import numpy as onp

from stadion.core import make_target_standardizer, make_sergio_standardizer

from stadion.synthetic import synthetic_sde_data
from stadion.scm import synthetic_scm_data
from assets.sergio.sergio import simulate as sergio_sampler

def make_data(*, seed, config):

    meta_data = dict()

    if "synth-sde-linear" == config["id"]:
        (train_tars, train_log), (test_tars, _) = synthetic_sde_data(seed, config=config)
        standardizer = make_target_standardizer(train_tars.data[0])
        
        # remember eigenvalue information for analysis
        mat = train_tars.true_param[0]
        meta_data["mat_eigenvals"] = onp.sort(onp.real(onp.linalg.eigvals(mat)))

        # remember simulation logs for analysis
        for k, v in train_log.items():
            meta_data[f"sde_sim_{k}"] = v

    elif "synth-sde-linear-raw" == config["id"]:
        (train_tars, train_log), (test_tars, _) = synthetic_sde_data(seed, config=config)
        standardizer = make_target_standardizer(train_tars.data[0], ignore=True)

        # remember eigenvalue information for analysis
        mat = train_tars.true_param[0]
        meta_data["mat_eigenvals"] = onp.sort(onp.real(onp.linalg.eigvals(mat)))

        # remember simulation logs for analysis
        for k, v in train_log.items():
            meta_data[f"sde_sim_{k}"] = v

    elif "synth-scm-linear" == config["id"]:
        train_tars, test_tars = synthetic_scm_data(seed, config=config)
        standardizer = make_target_standardizer(train_tars.data[0])

    elif "synth-scm-linear-raw" == config["id"]:
        train_tars, test_tars = synthetic_scm_data(seed, config=config)
        standardizer = make_target_standardizer(train_tars.data[0], ignore=True)

    elif "sergio" == config["id"]:
        train_tars, test_tars = sergio_sampler(seed, config=config)
        standardizer = make_target_standardizer(train_tars.data[0])

    else:
        raise KeyError(f"Invalid data_id `{config['id']}`")

    # extract meta data for analysis
    meta_data["mean"] = [x.mean(-2) for x in train_tars.data]
    meta_data["std"] = [x.std(-2) for x in train_tars.data]
    for k in meta_data.keys():
        if k == "var_names":
            continue
        if all([x.shape == meta_data[k][0].shape and x.ndim == 2 for x in meta_data[k]]):
            meta_data[k] = onp.stack(meta_data[k])

    # standardize variable-wise based on observational data
    train_tars = standardizer(train_tars)
    test_tars = standardizer(test_tars)

    return train_tars, test_tars, meta_data