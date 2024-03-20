import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal for")
warnings.filterwarnings("ignore", message="entering retry loop")
warnings.filterwarnings("ignore", message="Explicitly requested dtype") # shampoo float64 warning
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # hide gpus to tf to avoid OOM and cuda errors by conflicting with jax

import os
import shutil
import subprocess
from functools import partial
import time
import datetime
import traceback
from argparse import Namespace

import wandb
import jax
from jax import random, vmap
import jax.numpy as jnp
import numpy as onp

import optax
from optax._src import linear_algebra

from sklearn.decomposition import PCA

from stadion.core import sample_dynamical_system, Data
from stadion.plot import plot
from stadion.data import make_dataset, Batch, sample_batch_jax
from stadion.intervention import search_intv_theta_shift
from stadion.definitions import cpu_count, IS_CLUSTER, CONFIG_DIR

from stadion.sample import make_data

from stadion.utils.tree import tree_isnan
from stadion.utils.opt import update_ave, retrieve_ave, aggregate_scan
from stadion.utils.metrics import rbf_kernel, make_mse, make_wasserstein, is_unstable
from stadion.utils.partial import wrapped_mean
from stadion.utils.parse import load_config
from stadion.utils.version_control import get_gpu_info, get_gpu_info2


# time in min for syncing and wrapping up if job ends
SYNC_TIME_MIN = 20


def run_algo_wandb(wandb_config=None, eval_mode=False):
    """Function run by wandb.agent()"""

    # job setup
    t_init = time.time()
    exception_after_termination = None

    # wandb setup
    with wandb.init(**wandb_config):

        try:
            # this config will be set by wandb
            config = wandb.config

            # summary metrics we are interested in
            wandb.define_metric("loss", summary="min")
            wandb.define_metric("loss_fit", summary="min")
            wandb.define_metric("mse_train", summary="min")
            wandb.define_metric("mse_test", summary="min")
            wandb.define_metric("wasser_train", summary="min")
            wandb.define_metric("wasser_test", summary="min")

            wandb_run_dir = os.path.abspath(os.path.join(wandb.run.dir, os.pardir))
            print("wandb directory: ", wandb_run_dir, flush=True)

            """++++++++++++++   Data   ++++++++++++++"""
            jnp.set_printoptions(precision=2, suppress=True, linewidth=200)

            print("\nSimulating data...", flush=True)

            # load or sample data
            data_config = load_config(CONFIG_DIR / config.data_config, abspath=True)
            train_targets, test_targets, meta_data = make_data(seed=config.seed, config=data_config)

            print("done.\n", flush=True)

            """++++++++++++++   Run algorithm   ++++++++++++++"""

            _ = run_algo(train_targets, test_targets, config=config, eval_mode=eval_mode, t_init=t_init)


        except Exception as e:
            print("\n\n" + "-" * 30 + "\nwandb sweep exception traceback caught:\n")
            print(traceback.print_exc(), flush=True)
            exception_after_termination = e

        print("End of wandb.init context.\n", flush=True)

    # manual sync of offline wandb run -- this is for some reason much faster than the online mode of wandb
    print("Starting wandb sync ...", flush=True)
    subprocess.call(f"wandb sync {wandb_run_dir}", shell=True)

    # clean up
    try:
        shutil.rmtree(wandb_run_dir)
        print(f"deleted wandb directory after successful sync: `{wandb_run_dir}`", flush=True)
    except OSError as e:
        print("wandb dir not deleted.", flush=True)

    if exception_after_termination is not None:
        raise exception_after_termination

    print(f"End of run_algo_wandb after total walltime: "
          f"{str(datetime.timedelta(seconds=round(time.time() - t_init)))}",
          flush=True)



def run_algo(train_targets, test_targets, config=None, eval_mode=False, t_init=None):

    jnp.set_printoptions(precision=2, suppress=True, linewidth=200)
    key = random.PRNGKey(config.seed)
    t_run_algo = time.time()

    device_count = jax.device_count()
    local_device_count = jax.local_device_count()
    print(f"jax backend:   {jax.default_backend()} ")
    print(f"devices:       {device_count}")
    print(f"local_devices: {local_device_count}")
    print(f"cpu_count:     {cpu_count}", flush=True)
    print(f"gpu_info:      {get_gpu_info()}", flush=True)
    print(f"               {get_gpu_info2()}", flush=True)

    if type(config) == wandb.Config:
        # wandb case
        config.update(dict(d=train_targets.data[0].shape[-1]))
    else:
        # Namespace case
        config.d = train_targets.data[0].shape[-1]

    """++++++++++++++   Low-dim visualization   ++++++++++++++"""

    if not eval_mode and config.plot_proj:
        print("Computing projection transform based on target data...")
        all_target_data = []
        for data_env in train_targets.data:
            all_target_data.append(data_env)

        # # pca
        all_target_data = onp.concatenate(all_target_data, axis=0)
        proj = PCA(n_components=2, random_state=config.seed)
        _ = proj.fit(all_target_data)
        print("done.", flush=True)

    else:
        proj = None


    """++++++++++++++   Model and parameter initialization   ++++++++++++++"""
    # theta
    key, subk = random.split(key)

    if config.model == "linear_u":
        from stadion.models.linear_u import (
            f, sigma, init_theta, init_intv_theta, group_lasso, weighted_group_lasso,
        )
        theta = init_theta(subk, config.d, scale=config.init_scale, init_diag=config.init_diag,
                           force_linear_diag=config.force_linear_diag)

    elif config.model == "linear_u_diag":
        from stadion.models.linear_u_diag import (
            f, sigma, init_theta, init_intv_theta, group_lasso, weighted_group_lasso,
        )
        theta = init_theta(subk, config.d, scale=config.init_scale, init_diag=config.init_diag,
                           force_linear_diag=config.force_linear_diag)

    elif config.model == "lnl_u_diag":
        from stadion.models.lnl_u_diag import (
            f, sigma, init_theta, init_intv_theta, group_lasso,
        )
        f = partial(f, activation=config.mlp_activation)

        theta = init_theta(subk, config.d, hidden_size=config.mlp_hidden,
                           scale=config.init_scale, distribution=config.mlp_init,
                           force_linear_diag=config.force_linear_diag)

    else:
        raise KeyError(f"Unknown model `{config.model}`")


    # intv theta
    key, subk = random.split(key)
    intv_theta_train = init_intv_theta(subk, train_targets.intv.shape[0], config.d, scale_param=False, scale=config.init_scale)

    # empirical means
    print("train means masked by intv")
    print(jnp.array([data.mean(-2) for data in train_targets.data]) * train_targets.intv)

    if config.init_intv_at_mean:
        print(f"init intv_theta shift at empirical means")
        train_emp_means = jnp.array([data.mean(-2) for data in train_targets.data])
        intv_theta_train["shift"] += jnp.where(train_targets.intv, (train_emp_means - train_emp_means[0][None]), jnp.array(0))
        print("intv_theta_train: shift")
        print(intv_theta_train["shift"])
        print()

    """++++++++++++++   Kernel   ++++++++++++++"""

    def kernel(x_in, y_in, *args):
        # kernel function satisfying the same signature as f and sigma
        return rbf_kernel(x_in, y_in, jnp.array(config.bandwidth))


    """++++++++++++++   Data   ++++++++++++++"""

    print(f"Initializing dataset...", flush=True)
    assert config.auto_diff, "Current sampler only supports auto-diff"
    jax_train_targets = Data(**{k: jnp.array(v) for k, v in train_targets._asdict().items()})
    train_loader = partial(sample_batch_jax,
                           targets=jax_train_targets,
                           batch_size=config.batch_size,
                           batch_size_env=config.batch_size_env)

    print(f"done.", flush=True)

    """++++++++++++++   Auto-diff generator loss function   ++++++++++++++"""

    def generator(h, arg):
        def h_out(*z):
            fx = f(z[arg], *z[2:]) + config.reg_eps * z[arg]
            sx = sigma(z[arg], *z[2:])
            return fx @ jax.grad(h, arg)(*z) + 0.5 * jnp.trace(sx @ sx.T @ jax.hessian(h, arg)(*z))
        return h_out


    @partial(wrapped_mean, axis=(0, 1))
    @partial(vmap, in_axes=(None, None, Batch(x=0)), out_axes=0)
    @partial(vmap, in_axes=(None, None, Batch(y=0)), out_axes=0)
    def auto_loss_fun(param, intv_param, batch):
        """
        Computes generator loss function for each x,x' pair in data distribution using autodifferentation
        and the operator view of the generator.
        Needed to learn sigma.
        Then takes mean over shape [batch_size, batch_size]
        """
        assert batch.x.shape == batch.y.shape == batch.intv.shape == (config.d,)
        return generator(generator(kernel, 0), 1)(batch.x, batch.y, param, intv_param, batch.intv)


    """++++++++++++++   Cached generator loss function   ++++++++++++++"""

    @partial(wrapped_mean, axis=(0, 1))
    @partial(vmap, in_axes=(None, None, Batch(x=0, k0=0, k1=0, k2=0, k3=0)), out_axes=0)
    @partial(vmap, in_axes=(None, None, Batch(y=0, k0=0, k1=0, k2=0, k3=0)), out_axes=0)
    def cached_loss_fun(param, intv_param, batch):
        """
        Computes generator loss function for each x,x' pair in data distribution using cached kernel terms.
        Assumes sigma = identity.
        Then takes mean over shape [batch_size, batch_size]
        """
        assert batch.x.shape == batch.y.shape == batch.intv.shape == (config.d,)
        assert batch.k0.ndim == 2
        assert batch.k1.ndim == batch.k2.ndim == 1
        assert batch.k3.ndim == 0
        assert batch.k0.shape[0] == batch.k0.shape[1] == batch.k1.shape[0] == batch.k2.shape[0]

        # compute drift terms
        f_x = f(batch.x, param, intv_param, batch.intv)
        f_y = f(batch.y, param, intv_param, batch.intv)

        assert batch.x.shape == f_x.shape == f_y.shape

        # compute the four generator loss objective terms
        term0 = f_x @ batch.k0 @ f_y
        term1 = 0.5 * f_x @ batch.k1
        term2 = 0.5 * f_y @ batch.k2
        term3 = 0.25 * batch.k3

        assert term0.ndim == term1.ndim == term2.ndim == term3.ndim == 0

        return term0 + term1 + term2 + term3


    """++++++++++++++  Loss function   ++++++++++++++"""

    # select loss function according to whether we cached the kernel terms or whether we compute them with autodiff
    if config.auto_diff:
        loss_fun = auto_loss_fun
    else:
        loss_fun = cached_loss_fun


    @partial(vmap, in_axes=(None, None, 0), out_axes=0)
    def mean_loss(param, intv_param, batch):
        """
        For each env/target distribution individually, computes average of
        `L_x L_y kernel(x,y)` for each pair(x,y)~tar0,tar1
        """
        # select intv_param corresponding to the environment in the batch `batch`
        # by taking dot-product with environment one-hot indicator vector
        env_intv_param = jax.tree_util.tree_map(lambda leaf: jnp.einsum("e,e...", batch.env_indicator, leaf), intv_param)

        # compute loss
        ave_loss = loss_fun(param, env_intv_param, batch)
        return ave_loss


    """++++++++++++++   Objectives   ++++++++++++++"""

    def total_objective(params, batch):
        """
        Computes loss over environment and takes mean over shape [batch_size_env,]
        Then combines loss with possible regularizers
        """
        theta_in, intv_theta_in = params

        # compute loss in each target env (interventional distribution)
        # [batch_size_env]
        loss_fit = mean_loss(theta_in, intv_theta_in, batch)
        loss_fit = jnp.mean(loss_fit, axis=0)

        # regularizer
        if config.reg_type == "glasso":
            reg = group_lasso(theta_in)
        elif config.reg_type == "weightglasso-001":
            reg = weighted_group_lasso(theta_in, 0.01)
        elif config.reg_type == "weightglasso-01":
            reg = weighted_group_lasso(theta_in, 0.1)
        elif config.reg_type == "weightglasso-1":
            reg = weighted_group_lasso(theta_in, 1.0)
        else:
            raise KeyError(f"Unknown regularizer {config.reg_type}")
        assert reg.ndim == 0

        # loss, aux info
        return loss_fit + config.reg_strength * reg, dict(loss_fit=loss_fit)


    update_fun = jax.value_and_grad(total_objective, 0, has_aux=True)


    """++++++++++++++   Optimizer and update step ++++++++++++++"""

    # theta
    optimizer_modules = []
    if config.grad_clip_val is not None:
        optimizer_modules.append(optax.clip_by_global_norm(config.grad_clip_val))
    if config.optimizer == "adam":
        optimizer_modules.append(optax.adam(config.learning_rate))
    elif config.optimizer == "adam-9":
        optimizer_modules.append(optax.adam(config.learning_rate, b2=0.9))
    elif config.optimizer == "yogi":
        optimizer_modules.append(optax.yogi(config.learning_rate))
    elif config.optimizer == "yogi-9":
        optimizer_modules.append(optax.yogi(config.learning_rate, b2=0.9))
    elif config.optimizer == "adagrad":
        optimizer_modules.append(optax.adagrad(config.learning_rate))
    elif config.optimizer == "adafactor":
        optimizer_modules.append(optax.adafactor(config.learning_rate))
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer}")

    optimizer = optax.chain(*optimizer_modules)

    def update_step(carry, _):
        """
        A single update step of the optimizer
        """
        loop_key, theta_in, intv_theta_in, opt_state_in = carry
        param = (theta_in, intv_theta_in)

        # sample new batch
        loop_key, loop_subk = random.split(loop_key)
        batch = train_loader(key=loop_subk)

        # compute gradient of objective
        (l, l_aux), dparam = update_fun(param, batch)

        if config.force_linear_diag:
            dtheta_in, dintv_theta_in = dparam

            if "w1" in dtheta_in:
                dtheta_in["w1"] = dtheta_in["w1"].at[jnp.diag_indices(config.d)].set(0.0)
            if "x1" in dtheta_in:
                dtheta_in["x1"] = dtheta_in["x1"].at[jnp.diag_indices(config.d)].set(0.0)
            if "y1" in dtheta_in:
                dtheta_in["y1"] = dtheta_in["y1"].at[jnp.diag_indices(config.d)].set(0.0)
            if "v1" in dtheta_in:
                dtheta_in["v1"] = dtheta_in["v1"].at[...].set(0.0)

            dparam = (dtheta_in, dintv_theta_in)

        # update parameters
        param_update, opt_state_in = optimizer.update(dparam, opt_state_in, param)
        param = optax.apply_updates(param, param_update)

        # logging
        dtheta, dintv_theta = dparam
        grad_norm = linear_algebra.global_norm(dtheta)
        intv_grad_norm = linear_algebra.global_norm(dintv_theta)
        nan_occurred = tree_isnan(dtheta) | tree_isnan(theta)
        nan_occurred_intv = tree_isnan(dintv_theta) | tree_isnan(intv_theta_train)
        aux = dict(loss=l,
                   **l_aux,
                   grad_norm=grad_norm,
                   intv_grad_norm=intv_grad_norm,
                   nan_occurred=nan_occurred,
                   nan_occurred_intv=nan_occurred_intv)

        carry = (loop_key, *param, opt_state_in)
        return carry, aux


    @partial(jax.jit, static_argnums=(0,))
    def update_loop(n_steps, init):
        """
        Runs `n_steps` of the optimizer step `update_step`
        """
        return jax.lax.scan(update_step, init, None, length=n_steps)


    """++++++++++++++   Optimization ++++++++++++++"""
    print("Starting inference...")

    # init optimizer
    opt_state = optimizer.init((theta, intv_theta_train))

    # sampling function
    # f_eps =  lambda x, *args: f(x, *args) - config.sampler_eps * x
    assert config.sampler_eps == 0.0
    f_eps =  lambda x, *args: f(x, *args)
    sampler = partial(sample_dynamical_system, config=Namespace(**config.sde), f=f_eps, sigma=sigma)

    if not eval_mode:
        # MSE
        mse_accuracy = make_mse(sampler=sampler, n=config.metric_batch_size)

        # wasserstein distance
        wasser_eps_train = jnp.ones(len(train_targets.data)) * 10.
        wasser_eps_test = jnp.ones(len(test_targets.data)) * 10.

        wasserstein_accuracy_train = make_wasserstein(wasser_eps_train, sampler=sampler, n=config.metric_batch_size)
        wasserstein_accuracy_test = make_wasserstein(wasser_eps_test, sampler=sampler, n=config.metric_batch_size)

    # loop
    t_loop = time.time()

    # sample new batch
    for t_super in range(int(config.steps / config.log_every)):
        t = (t_super + 1) * config.log_every

        # update loop for `log_every` steps
        (key, theta, intv_theta_train, opt_state), loss_aux = \
            update_loop(config.log_every, (key, theta, intv_theta_train, opt_state))

        # log averages
        mean_aux = aggregate_scan(loss_aux)
        if "w1" in theta and theta["w1"].ndim == 2 and theta["w1"].shape[0] == theta["w1"].shape[1]:
            mean_aux["max_eigenvalue"] = jnp.real(onp.linalg.eigvals(theta["w1"])).max()
        else:
            mean_aux["max_eigenvalue"] = jnp.array(jnp.nan)

        if eval_mode:
            t_elapsed = time.time() - t_loop
            t_loop = time.time()
            print(f"t: {t: >5d} "
                  f"loss: {mean_aux['loss']: >12.6f}  "
                  f"gnorm: {mean_aux['grad_norm']: >6.4f}  "
                  f"eigv: {mean_aux['max_eigenvalue']: >4.2f}  "
                  f"min: {(config.steps - t) * t_elapsed / config.log_every / 60.0: >4.1f}  "
                  , flush=True)

            continue

        """
        ------------------------------------
        Evaluation and logging 
        Only done when not in eval_mode
        ------------------------------------
        """

        # check for early termination due to cluster job time
        terminate_on_this_one = t_init is not None and (time.time() - t_init) > (config.cluster_t_max - SYNC_TIME_MIN) * 60

        # logging
        wandb_dict = {}

        if t != 0 or terminate_on_this_one:

            # time
            t_elapsed = time.time() - t_loop
            t_loop = time.time()

            # wandb
            log_dict = {
                "loss": mean_aux["loss"].item(),
                "loss_fit": mean_aux["loss_fit"].item(),
                "grad_norm": mean_aux["grad_norm"].item(),
                "intv_grad_norm": mean_aux["intv_grad_norm"].item(),
                "max_eigenvalue": mean_aux['max_eigenvalue'].item(),
                "t_step": t_elapsed / config.log_every,
            }

            # eval metrics
            key, subk = random.split(key)
            log_dict["mse_train"], _ = \
                mse_accuracy(subk, train_targets, theta, intv_theta_train)

            if IS_CLUSTER and (((not t % int(config.log_long_every)) and t != 0) or terminate_on_this_one):
                key, subk = random.split(key)
                log_dict["wasser_train"], _ = \
                    wasserstein_accuracy_train(subk, train_targets, theta, intv_theta_train)

                # ------------ debugging ------------
                key, subk = random.split(key)
                debug_samples = sampler(subk, theta, intv_theta_train, train_targets.intv, n_samples=config.metric_batch_size)
                log_dict["unstable_observ_100"] = is_unstable(debug_samples, std_bound=100)[0].item()
                # -----------------------------------

            wandb_dict.update(**log_dict)
            print(f"t: {t: >5d} "
                  f"loss: {log_dict['loss']: >12.6f}  "
                  # f"mse: {log_dict['mse_train']: >8.3f}  "
                  # f"gnorm: {log_dict['grad_norm']: >6.4f}  "
                  f"eigv: {log_dict['max_eigenvalue']: >4.2f}  "
                  f"min: {(config.steps - t) * log_dict['t_step'] / 60.0: >4.1f}  "
                  f"sec/step: {log_dict['t_step']: >4.2f} "
                  , flush=True)


        if (((not t % int(config.plot_every) or not t % int(config.eval_every)) and t != 0)
            or t == config.steps
            or terminate_on_this_one):

            assert test_targets is not None

            # assumed information about test targets
            test_target_intv = test_targets.intv
            test_emp_means = test_target_intv * jnp.array([data.mean(-2) for data in test_targets.data])

            # init new intv_theta_test with scale 0.0
            key, subk = random.split(key)
            intv_theta_test_init = init_intv_theta(subk, test_target_intv.shape[0], config.d,
                                                   scale_param=config.learn_intv_scale, scale=0.0)

            # update estimate of intervention effects in test set
            key, subk = random.split(key)
            intv_theta_test, logs = search_intv_theta_shift(subk, theta=theta,
                                                            intv_theta=intv_theta_test_init,
                                                            target_means=test_emp_means,
                                                            target_intv=test_target_intv,
                                                            sampler=sampler,
                                                            n_samples=config.metric_batch_size)

            # to compute metrics, use test data
            key, subk = random.split(key)
            logs["mse_test"], _ = \
                mse_accuracy(subk, test_targets, theta, intv_theta_test)

            if IS_CLUSTER:
                key, subk = random.split(key)
                logs["wasser_test"], _= \
                    wasserstein_accuracy_test(subk, test_targets, theta, intv_theta_test)

            wandb_dict.update(**logs)

            print(f"t: {t:5d} "
                  f"fMSE_test: {logs['mse_test']: >8.3f}  "
                  # f"iMSE_test: {logs['imse_test']: >8.3f}  "
                  , flush=True)

            if not t % int(config.plot_every) or t == config.steps or terminate_on_this_one:

                for plot_suffix, plot_tars, plot_intv_theta in [("train", train_targets, intv_theta_train),
                                                                ("test", test_targets, intv_theta_test)]:

                    # simulate rollouts
                    key, subk = random.split(key)
                    samples, trajs_full, _ = sampler(subk, theta, plot_intv_theta, plot_tars.intv,
                                                     n_samples=config.metric_batch_size, return_traj=True)

                    assert samples.shape[1] >= config.plot_batch_size, "Error: should sample at least `plot_batch_size` samples "
                    samples, trajs_full_single = samples[:, :config.plot_batch_size, :], trajs_full[:, 0]

                    # # plot with batched target
                    # key, subk = random.split(key)
                    # batched_plot_tars = sample_subset(subk, plot_tars, config.plot_batch_size)

                    wandb_images = plot(samples, trajs_full_single, plot_tars,
                                        # title_prefix=f"{plot_suffix} t={t} ",
                                        title_prefix=f"{plot_suffix}",
                                        theta=theta,
                                        intv_theta=plot_intv_theta,
                                        true_param=train_targets.true_param,
                                        ref_data=train_targets.data[0],
                                        cmain="grey",
                                        cfit="blue",
                                        cref="grey",
                                        t_current=t,
                                        # plot_mat=False,
                                        # plot_mat=True,
                                        plot_mat=plot_suffix == "train",
                                        # plot_mat=plot_suffix== "train" and t == int(config.steps),
                                        # plot_params=False,
                                        plot_params=plot_suffix== "train" and IS_CLUSTER,
                                        # plot_params=True,
                                        plot_acorr=False,
                                        # proj=None,
                                        proj=proj,
                                        # proj=proj,
                                        # proj=proj if plot_suffix == "train" else None,
                                        # proj=proj if t == config.steps else None,
                                        # proj=proj if t == config.steps and plot_suffix == "train" else None,
                                        # plot_intv_marginals=False,
                                        # plot_intv_marginals=True,
                                        plot_intv_marginals=IS_CLUSTER,
                                        # plot_intv_marginals=not IS_CLUSTER or (plot_suffix == "train"),
                                        # plot_intv_marginals=plot_suffix == "train",
                                        # plot_pairwise_grid=False,
                                        # plot_pairwise_grid=True,
                                        # plot_pairwise_grid=not IS_CLUSTER or (plot_suffix == "train"),
                                        # plot_pairwise_grid=plot_suffix== "train",
                                        # plot_pairwise_grid=plot_suffix== "test",
                                        plot_pairwise_grid=plot_suffix== "train" or (t == config.steps and IS_CLUSTER),
                                        # plot_pairwise_grid=plot_suffix== "train" and t == config.steps and config.d <= 5,
                                        grid_type="hist-kde",
                                        # contours=(0.68, 0.95),
                                        # contours_alphas=(0.33, 1.0),
                                        contours=(0.90,),
                                        contours_alphas=(1.0,),
                                        # scotts_bw_scaling=0.75,
                                        scotts_bw_scaling=1.0,
                                        size_per_var=1.0,
                                        plot_max=config.plot_max,
                                        to_wandb=IS_CLUSTER)


                    if wandb_images:
                        wandb_images = {f"{k}-{plot_suffix}" if "matrix" not in k else k: v
                                        for k, v in wandb_images.items()}
                        wandb_dict.update(wandb_images)


        if wandb_dict:
            if t == config.steps:
                wandb_dict["nan_occurred"] = int(False)

            assert not eval_mode
            wandb.log(wandb_dict, step=t + 1)

        if terminate_on_this_one:
            print("Exiting early due to time constraint.", flush=True)
            break

    print(f"End of run_algo after total walltime: "
          f"{str(datetime.timedelta(seconds=round(time.time() - t_run_algo)))}",
          flush=True)

    return sampler, init_intv_theta, theta, intv_theta_train


if __name__ == "__main__":
    debug_config = Namespace()

    # fixed
    debug_config.seed = 0

    # data
    debug_config.data_config = "dev/linear.yaml"
    # debug_config.data_config = "dev/sergio.yaml"

    # simulation (for prediction only)
    debug_config.sde = dict(
        method="explicit",
        rollout_restarts=10,
        n_samples_burnin=100,
        thinning=1000,
        dt=0.01,
    )

    # model
    debug_config.model = "linear_u_diag"
    # debug_config.model = "lnl_u_diag"

    debug_config.sampler_eps = 0.0
    debug_config.reg_eps = 0.0

    debug_config.init_scale = 0.001
    debug_config.init_diag = 0.0
    debug_config.init_intv_at_mean = True
    debug_config.learn_intv_scale = False
    debug_config.mlp_hidden = 4
    debug_config.mlp_activation = "tanh"
    debug_config.mlp_init = "uniform"
    debug_config.auto_diff = True

    # optimization
    debug_config.batch_size = 192
    debug_config.batch_size_env = 1
    debug_config.bandwidth = 1.0
    debug_config.reg_strength = 0.01
    debug_config.reg_type = "glasso"
    debug_config.grad_clip_val = None
    debug_config.force_linear_diag = True

    debug_config.steps = 2000
    debug_config.optimizer = "adam"
    debug_config.learning_rate = 0.003

    debug_config.log_every = 100
    debug_config.eval_every = 10000
    debug_config.log_long_every = 10000
    debug_config.plot_every = 10000
    debug_config.plot_max = 3
    debug_config.metric_batch_size = 1024
    debug_config.plot_batch_size = 384
    debug_config.plot_proj = False

    debug_config.cluster_t_max = 1000

    debug_wandb_config = dict(config=debug_config, mode="disabled")
    run_algo_wandb(wandb_config=debug_wandb_config, eval_mode=False)

