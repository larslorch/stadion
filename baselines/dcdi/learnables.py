"""
GraN-DAG

Copyright © 2019 Sébastien Lachapelle, Philippe Brouillard, Tristan Deleu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
import sys
import math
sys.path.insert(0, '/')

import warnings
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


try:
    from .base_model import BaseModel
except ImportError:
    from base_model import BaseModel


class LearnableModel(BaseModel):
    def __init__(self, num_vars, num_layers, hid_dim, num_params,
                 nonlin="leaky-relu", intervention=False,
                 intervention_type="perfect",
                 intervention_knowledge="known", num_regimes=1):

        super(LearnableModel, self).__init__(num_vars, num_layers, hid_dim, num_params,
                                             nonlin=nonlin,
                                             intervention=intervention,
                                             intervention_type=intervention_type,
                                             intervention_knowledge=intervention_knowledge,
                                             num_regimes=num_regimes)
        self.reset_params()

    def compute_log_likelihood(self, x, weights, biases, extra_params,
                               detach=False, mask=None, regime=None):
        """
        Return log-likelihood of the model for each example.
        WARNING: This is really a joint distribution only if the DAGness constraint on the mask is satisfied.
                 Otherwise the joint does not integrate to one.
        :param x: (batch_size, num_vars)
        :param weights: list of tensor that are coherent with self.weights
        :param biases: list of tensor that are coherent with self.biases
        :param mask: tensor, shape=(batch_size, num_vars)
        :param regime: np.ndarray, shape=(batch_size,)
        :return: (batch_size, num_vars) log-likelihoods
        """
        density_params = self.forward_given_params(x, weights, biases, mask, regime)


        if len(extra_params) != 0:
            extra_params = self.transform_extra_params(self.extra_params)
        log_probs = []

        for i in range(self.num_vars):
            density_param = list(torch.unbind(density_params[i], 1))
            if len(extra_params) != 0:
                density_param.extend(list(torch.unbind(extra_params[i], 0)))
            conditional = self.get_distribution(density_param)
            x_d = x[:, i].detach() if detach else x[:, i]
            log_probs.append(conditional.log_prob(x_d).unsqueeze(1))

        return torch.cat(log_probs, 1)

    def get_distribution(self, dp):
        raise NotImplementedError

    def transform_extra_params(self, extra_params):
        raise NotImplementedError

    def sampler(self, intv_theta, intv, *, n_samples, g, toporder,
                weights, biases, extra_params, weight_sel):
        """
        Sample `n_samples` from the model given `intv_theta` information.
        Since this requires a topological ordering, it will randomly break cycles in case the learned graph is not a DAG.
        Returns:
            [n_samples, num_vars]
        """

        with torch.no_grad():

            d = g.shape[-1]

            # extract shift intervention parameters
            shift = intv_theta["shift"] * intv
            assert intv_theta["shift"].shape == intv.shape

            n_envs = intv.shape[0]
            assert intv.shape == (n_envs, d), f"{intv.shape}"

            # extra_params: [d,] (std of gaussian)
            if len(extra_params) != 0:
                extra_params = self.transform_extra_params(self.extra_params)
                assert len(extra_params) == d
                assert extra_params[0].shape == (1,)

            x = np.zeros((n_envs, n_samples, d))

            # regular ancestral sampling
            for j in toporder:

                # compute params of conditional distribution of x_j
                # we do a full forward pass through MLP and just use the dimension j we care about

                # env_density_params: length-`d` tuple of [..., num_params], which are params of each variable conditional
                # here: num_params = 1 (mean of gaussian)
                env_density_params = self.forward_given_params_eval(torch.tensor(x), weights, biases, torch.tensor(g),
                                                                    weight_sel=weight_sel)
                assert len(env_density_params) == d
                assert env_density_params[0].shape == (n_envs, n_samples, 1)

                env_density_param_j = list(torch.unbind(env_density_params[j], -1))
                if len(extra_params) != 0:
                    env_density_param_j.extend(list(torch.unbind(extra_params[j], 0)))
                assert len(env_density_param_j) == 2

                # sample from conditional of x_j
                conditional_j = self.get_distribution(env_density_param_j)
                x_j = conditional_j.sample()

                assert x_j.shape == x[:, :, j].shape

                # apply shift intervention if passed
                x_j += shift[:, j, None]

                # assert that the field we will fill has not been filled yet (as we do ancestral sampling over a DAG)
                assert np.array_equal(x[:, :, j], np.zeros((n_envs, n_samples))), \
                    f"We do ancestral sampling over a DAG. The dimension we currently sample should not be filled yet."

                # insert observations
                x[:, :, j] = x_j

        return x



class LearnableModel_NonLinGaussANM(LearnableModel):
    def __init__(self, num_vars, num_layers, hid_dim, nonlin="leaky-relu",
                 intervention=False,
                 intervention_type="perfect",
                 intervention_knowledge="known",
                 num_regimes=1):
        super(LearnableModel_NonLinGaussANM, self).__init__(num_vars, num_layers, hid_dim, 1, nonlin=nonlin,
                                                            intervention=intervention,
                                                            intervention_type=intervention_type,
                                                            intervention_knowledge=intervention_knowledge,
                                                            num_regimes=num_regimes)
        # extra parameters are log_std
        extra_params = np.ones((self.num_vars,))
        np.random.shuffle(extra_params)
        # each element in the list represents a variable, the size of the element is the number of extra_params per var
        self.extra_params = nn.ParameterList()
        for extra_param in extra_params:
            self.extra_params.append(nn.Parameter(torch.tensor(np.log(extra_param).reshape(1)).type(torch.Tensor)))

    def get_distribution(self, dp):
        return torch.distributions.normal.Normal(dp[0], dp[1])

    def transform_extra_params(self, extra_params):
        transformed_extra_params = []
        for extra_param in extra_params:
            transformed_extra_params.append(torch.exp(extra_param))
        return transformed_extra_params  # returns std_dev
