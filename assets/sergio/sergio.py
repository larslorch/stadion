import sys
import numpy as np
import jax.numpy as jnp
from jax import random
import math
import copy
from collections import defaultdict
import itertools

from stadion.core import Data
from stadion.synthetic import erdos_renyi, scale_free, sbm


class Gene(object):

    def __init__(self, geneID, geneType, binID = -1):

        """
        geneType: 'MR' master regulator or 'T' target
        bindID is optional
        """

        self.ID = geneID
        self.Type = geneType
        self.binID = binID
        self.Conc = []
        self.simulatedSteps_ = 0

    def append_Conc (self, currConc):
        if isinstance(currConc, list):
            if currConc[0] < 0.0:
                self.Conc.append([0.0])
            else:
                self.Conc.append(currConc)
        else:
            if currConc < 0.0:
                self.Conc.append(0.0)
            else:
                self.Conc.append(currConc)

    def incrementStep (self):
        self.simulatedSteps_ += 1

    def set_scExpression(self, list_indices):
        """
        selects input indices from self.Conc and form sc Expression
        """
        self.scExpression = np.array(self.Conc)[list_indices]


class Sergio:

    def __init__(self,
                 rng,
                 number_genes,
                 number_bins,
                 number_sc,
                 noise_params,
                 noise_type,
                 decays,
                 knockout_target=None,
                 knockout_multiplier=None,
                 reference_meanExpression=None,
                 sampling_state = 10,
                 tol = 1e-3,
                 window_length = 100,
                 dt = 0.01,
                 k_coop=None,
                 safety_steps=0):
        """
        Noise is a gaussian white noise process with zero mean and finite variance.
        noise_params: The amplitude of noise in CLE. This can be a scalar to use
        for all genes or an array with the same size as number_genes.
        Tol: p-Value threshold above which convergence is reached
        window_length: length of non-overlapping window (# time-steps) that is used to realize convergence
        dt: time step used in  CLE
        noise_params and decays: Could be an array of length number_genes, or single value to use the same value for all genes
        number_sc: number of single cells for which expression is simulated
        sampling_state (>=1): single cells are sampled from sampling_state * number_sc steady-state steps
        noise_type: We consider three types of noise, 'sp': a single intrinsic noise is associated to production process, 'spd': a single intrinsic noise is associated to both
        production and decay processes, 'dpd': two independent intrinsic noises are associated to production and decay processes
        """

        self.rng = rng

        self.nGenes_ = number_genes
        self.nBins_ = number_bins
        self.nSC_ = number_sc
        self.sampling_state_ = sampling_state
        self.tol_ = tol
        self.winLen_ = window_length
        self.dt_ = dt
        self.safety_steps = safety_steps
        self.level2verts_ = {}
        self.gID_to_level_and_idx = {} # This dictionary gives the level and idx in self.level2verts_ of a given gene ID
        self.maxLevels_ = 0
        self.init_concs_ = np.zeros((number_genes, number_bins))
        self.meanExpression = -1 * np.ones((number_genes, number_bins))
        self.noiseType_ = noise_type
        self.nConvSteps = np.zeros(number_bins) # This holds the number of simulated steps till convergence

        if knockout_target is None:
            knockout_target = np.zeros(number_genes).astype(bool)
        self.knockout_target = knockout_target

        if knockout_multiplier is None:
            knockout_multiplier = np.ones(number_genes)
        self.knockout_multiplier = knockout_multiplier

        self.reference_meanExpression = reference_meanExpression
        self.k_coop = k_coop

        ############
        # This graph stores for each vertex: parameters(interaction
        # parameters for non-master regulators and production rates for master
        # regulators), tragets, regulators and level
        ############
        self.graph_ = {}

        if np.isscalar(noise_params):
            self.noiseParamsVector_ = np.repeat(noise_params, number_genes)
        elif np.shape(noise_params)[0] == number_genes:
            self.noiseParamsVector_ = noise_params
        else:
            print ("Error: expect one noise parameter per gene")


        if np.isscalar(decays) == 1:
            self.decayVector_ = np.repeat(decays, number_genes)
        elif np.shape(decays)[0] == number_genes:
            self.decayVector_ = decays
        else:
            print ("Error: expect one decay parameter per gene")
            sys.exit()


    def custom_graph(self, *, g, k, b, hill):
        """
        Prepare custom graph model and coefficients
        Args:
            g: [nGenes_, nGenes_] graph
            k: [nGenes_, nGenes_] interaction coeffs
            b: [nGenes_, nBins_] basal reproduction rate of source nodes (master regulators)
            hill: [nGenes_, nGenes_] hill coefficients of nonlinear interactions
        """

        # check inputs
        assert g.shape == k.shape
        assert g.shape == hill.shape
        assert g.shape[0] == self.nGenes_
        assert g.shape[1] == self.nGenes_
        assert b.shape[0] == self.nGenes_
        assert b.shape[1] == self.nBins_
        assert np.allclose(g[np.diag_indices(g.shape[0])], 0.0), f"No self loops allowed"
        self.hill_default = np.mean(hill)

        # following steps of original function
        for i in range(self.nGenes_):
            self.graph_[i] = {}
            self.graph_[i]['targets'] = []


        self.master_regulators_idx_ = set()

        for j in range(self.nGenes_):

            is_parent = g[:, j]

            # master regulator (no parents)
            if is_parent.sum() == 0:

                self.master_regulators_idx_.add(j)
                self.graph_[j]['rates'] = b[j]
                self.graph_[j]['regs'] = []
                self.graph_[j]['level'] = -1

            # regular gene (target)
            else:

                currInteraction = []
                currParents = []
                for u in np.where(is_parent == 1)[0]:
                    currInteraction.append((u, k[u, j], hill[u, j], 0))  # last zero shows half-response, it is modified in another method
                    currParents.append(u)
                    self.graph_[u]['targets'].append(j)

                self.graph_[j]['params'] = currInteraction
                self.graph_[j]['regs'] = currParents
                self.graph_[j]['level'] = -1  # will be modified later


        self.find_levels_(self.graph_)

    def find_levels_ (self, graph):
        """
        # This is a helper function that takes a graph and assigns layer to all
        # verticies. It uses longest path layering algorithm from
        # Hierarchical Graph Drawing by Healy and Nikolovself. A bottom-up
        # approach is implemented to optimize simulator run-time. Layer zero is
        # the last layer for which expression are simulated
        # U: verticies with an assigned layer
        # Z: vertizies assigned to a layer below the current layer
        # V: set of all verticies (genes)

        This also sets a dictionary that maps a level to a matrix (in form of python list)
        of all genes in that level versus all bins

        Note to self:
        This is like DFS topsort, but compressing the length of levels as much as possible
        Essentially, root nodes have the highest level (to be simulated first) and sink nodes have level 0,
        and any node upstream of a node has higher level
        Sets:
            level2verts_
                {l: [M, bins] where M is number of genes in on level l}

            gID_to_level_and_idx
                {v: (level, j) where j in index of vertex v in `level2verts[level]` for vertex v in graph
        """

        U = set()
        Z = set()
        V = set(graph.keys())

        currLayer = 0
        self.level2verts_[currLayer] = []
        idx = 0

        while U != V:
            currVerts = set(filter(lambda v: set(graph[v]['targets']).issubset(Z), V-U))

            for v in currVerts:
                graph[v]['level'] = currLayer
                U.add(v)
                if {v}.issubset(self.master_regulators_idx_):
                    allBinList = [Gene(v,'MR', i) for i in range(self.nBins_)]
                    self.level2verts_[currLayer].append(allBinList)
                    self.gID_to_level_and_idx[v] = (currLayer, idx)
                    idx += 1
                else:
                    allBinList = [Gene(v,'T', i) for i in range(self.nBins_)]
                    self.level2verts_[currLayer].append(allBinList)
                    self.gID_to_level_and_idx[v] = (currLayer, idx)
                    idx += 1

            currLayer += 1
            Z = Z.union(U)
            self.level2verts_[currLayer] = []
            idx = 0

        self.level2verts_.pop(currLayer)
        self.maxLevels_ = currLayer - 1

        self.set_scIndices_()

    def set_scIndices_ (self):
        """
        # Set the single cell indices that are sampled from steady-state steps
        # Note that sampling should be performed from the end of Concentration list
        # Note that this method should run after building graph(and layering) and should
        be run just once!
        """

        # time indices when to collect "single-cell" expression snapshots
        self.scIndices_ = self.rng.integers(low = - self.sampling_state_ * self.nSC_, high = 0, size = self.nSC_)

    def calculate_required_steps_(self, level):
        """
        # Calculates the number of required simulation steps after convergence at each level.
        # safety_steps: estimated number of steps required to reach convergence (same), although it is not neede!
        """
        #TODO: remove this safety step

        # Note to self: as safety measure leaving this safety step to double check
        # that knockouts/knockdowns have reached steady-state
        # however, we do initialize the concentrations correctly so should be fine without it

        return self.sampling_state_ * self.nSC_ + level * self.safety_steps

    def calculate_half_response_(self, level):
        """
        Calculates the half response for all interactions between previous layer
        and current layer
        """

        currGenes = self.level2verts_[level]

        for g in currGenes: # g is list of all bins for a single gene
            c = 0
            if g[0].Type == 'T':
                for interTuple in self.graph_[g[0].ID]['params']:
                    regIdx = interTuple[0]

                    # if there is a reference mean expression, use that to set the half response parameter,
                    # so that we use the same parametric model in all interventional (knockout) settings
                    if self.reference_meanExpression is not None:
                        meanArr = self.reference_meanExpression[regIdx]
                    else:
                        meanArr = self.meanExpression[regIdx]

                    if set(meanArr) == set([-1]):
                        print ("Error: Something's wrong in either layering or simulation. Expression of one or more genes in previous layer was not modeled.")
                        sys.exit()

                    self.graph_[g[0].ID]['params'][c] = (self.graph_[g[0].ID]['params'][c][0],
                                                         self.graph_[g[0].ID]['params'][c][1],
                                                         self.graph_[g[0].ID]['params'][c][2],
                                                         np.mean(meanArr))
                    c += 1
            #Else: g is a master regulator and does not need half response

    def hill_(self, reg_conc, half_response, coop_state, repressive = False):
        """
        So far, hill function was used in the code to model 1 interaction at a time.
        So the inputs are single values instead of list or array. Also, it models repression based on this assumption.
        if decided to make it work on multiple interaction, repression should be taken care as well.
        """
        if reg_conc == 0:
            if repressive:
                return 1
            else:
                return 0
        else:
            if repressive:
                return 1 - np.true_divide(np.power(reg_conc, coop_state), (np.power(half_response, coop_state) + np.power(reg_conc, coop_state)) )
            else:
                return np.true_divide(np.power(reg_conc, coop_state), (np.power(half_response, coop_state) + np.power(reg_conc, coop_state)) )

    def init_gene_bin_conc_ (self, level):
        """
        Initilizes the concentration of all genes in the input level

        Note: calculate_half_response_ should be run before this method
        """

        currGenes = self.level2verts_[level]
        for g in currGenes:
            gID = g[0].ID

            # adjust initial concentration for possible knockout/knockdown experiment
            if self.knockout_target[gID]:
                interv_factor = self.knockout_multiplier[gID]

            else:
                assert self.knockout_multiplier[gID] == 1.0, f"Knockout multiplier should be 1 if not knocked out." \
                                                             f"Got: {self.knockout_multiplier[gID]}"
                interv_factor = 1.0

            # initialize at expected steady-state
            if g[0].Type == 'MR':
                allBinRates = self.graph_[gID]['rates']

                for bIdx, rate in enumerate(allBinRates):
                    g[bIdx].append_Conc(np.true_divide(interv_factor * rate, self.decayVector_[g[0].ID]))

            else:
                params = self.graph_[g[0].ID]['params']

                for bIdx in range(self.nBins_):
                    rate = 0
                    for interTuple in params:
                        meanExp = self.meanExpression[interTuple[0], bIdx]
                        rate += np.abs(interTuple[1]) * self.hill_(meanExp, interTuple[3], interTuple[2], interTuple[1] < 0)

                    # cooperative regulation
                    if self.k_coop is not None and len(params) > 1:
                        regulator_combinations = list(itertools.combinations(params, 2))
                        for interTuple0, interTuple1 in regulator_combinations:
                            meanExp0 = self.meanExpression[interTuple0[0], bIdx]
                            meanExp1 = self.meanExpression[interTuple1[0], bIdx]
                            k_coop = self.k_coop[g[0].ID, interTuple0[0], interTuple1[0]]
                            rate += np.abs(k_coop) * self.hill_(meanExp0 * meanExp1,
                                                                interTuple0[3] * interTuple1[3],
                                                                self.hill_default,
                                                                k_coop < 0)

                    g[bIdx].append_Conc(np.true_divide(interv_factor * rate, self.decayVector_[g[0].ID]))

    def calculate_prod_rate_(self, bin_list, level):
        """
        calculates production rates for the input list of gene objects in different bins but all associated to a single gene ID
        """
        type = bin_list[0].Type

        if (type == 'MR'):
            rates = self.graph_[bin_list[0].ID]['rates']
            return np.array([rates[gb.binID] for gb in bin_list])

        else:
            params = self.graph_[bin_list[0].ID]['params']
            Ks = [np.abs(t[1]) for t in params]
            regIndices = [t[0] for t in params]
            binIndices = [gb.binID for gb in bin_list]
            currStep = bin_list[0].simulatedSteps_
            hillMatrix = np.zeros((len(regIndices), len(binIndices)))

            for tupleIdx, rIdx in enumerate(regIndices):
                regGeneLevel = self.gID_to_level_and_idx[rIdx][0]
                regGeneIdx = self.gID_to_level_and_idx[rIdx][1]
                regGene_allBins = self.level2verts_[regGeneLevel][regGeneIdx]
                for colIdx, bIdx in enumerate(binIndices):
                    hillMatrix[tupleIdx, colIdx] = \
                        self.hill_(regGene_allBins[bIdx].Conc[currStep],
                                   params[tupleIdx][3],
                                   params[tupleIdx][2],
                                   params[tupleIdx][1] < 0)

            prod_rate = np.matmul(Ks, hillMatrix)

            # cooperative regulation
            if self.k_coop is not None and len(params) > 1:
                regulator_combinations = list(itertools.combinations(params, 2))
                K_coops = [self.k_coop[bin_list[0].ID, interTuple0[0], interTuple1[0]]
                           for interTuple0, interTuple1 in regulator_combinations]
                K_coops_abs = [np.abs(k) for k in K_coops]
                hillMatrix_coop = np.zeros((len(regulator_combinations), len(binIndices)))

                for tupleIdx, (interTuple0, interTuple1) in enumerate(regulator_combinations):

                    regGeneLevel0 = self.gID_to_level_and_idx[interTuple0[0]][0]
                    regGeneIdx0 = self.gID_to_level_and_idx[interTuple0[0]][1]
                    regGene_allBins0 = self.level2verts_[regGeneLevel0][regGeneIdx0]

                    regGeneLevel1 = self.gID_to_level_and_idx[interTuple1[0]][0]
                    regGeneIdx1 = self.gID_to_level_and_idx[interTuple1[0]][1]
                    regGene_allBins1 = self.level2verts_[regGeneLevel1][regGeneIdx1]

                    for colIdx, bIdx in enumerate(binIndices):
                        hillMatrix_coop[tupleIdx, colIdx] = \
                            self.hill_(regGene_allBins0[bIdx].Conc[currStep] * regGene_allBins1[bIdx].Conc[currStep],
                                       interTuple0[3] * interTuple1[3],
                                       self.hill_default,
                                       K_coops[tupleIdx] < 0)

                prod_rate += np.matmul(K_coops_abs, hillMatrix_coop)

            return prod_rate


    def CLE_simulator_(self, level):

        self.calculate_half_response_(level)
        self.init_gene_bin_conc_(level)
        nReqSteps = self.calculate_required_steps_(level)

        # list of lists of genes at the current level
        # each inner list contains `nBins` elements, representing expressions of genes of a given cell type (bin) at time t
        # level2verts_: {level: [[...], ...., [...]]}
        sim_set = np.copy(self.level2verts_[level]).tolist()

        # simulate expression of each gene at this level (vectorized for all bins)
        while sim_set != []:

            delIndicesGenes = []

            # g: list of gene objects of length 'nBins'
            for gi, bin_list in enumerate(sim_set):

                # gene id (row/col in adjacency matrix)
                gID = bin_list[0].ID

                # level in graph, index in list of expressions per bin (same as gi)
                gLevel, gIDX = self.gID_to_level_and_idx[gID]
                assert level == gLevel, "Levels should match"
                assert gi == gIDX, "index in gene-bin matrix should match"

                # [nBins,] current expressions
                currExp = np.array([gb.Conc[-1] for gb in bin_list])

                # [nBins,] production rate of gene given parents
                # adjust initial concentration for possible knockout/knockdown experiment
                if self.knockout_target[gID]:
                    prod_rate = self.knockout_multiplier[gID] * self.calculate_prod_rate_(bin_list, level)

                else:
                    assert self.knockout_multiplier[gID] == 1.0, f"Knockout multiplier should be 1 if not knocked out." \
                                                                 f"Got: {self.knockout_multiplier[gID]}"

                    prod_rate = self.calculate_prod_rate_(bin_list, level)


                # [nBins,] decay rate
                decay = np.multiply(self.decayVector_[gID], currExp)

                # sample noise
                if self.noiseType_ == 'sp':
                    # This notation is inconsistent with our formulation, dw should
                    # include dt^0.5 as well, but here we multipy dt^0.5 later
                    # [nBins, ]
                    dw = self.rng.normal(size = len(currExp))
                    amplitude = np.multiply (self.noiseParamsVector_[gID] , np.power(prod_rate, 0.5))
                    noise = np.multiply(amplitude, dw)

                elif self.noiseType_ == "spd":
                    # [nBins, ]
                    dw = self.rng.normal(size = len(currExp))
                    amplitude = np.multiply (self.noiseParamsVector_[gID] , np.power(prod_rate, 0.5) + np.power(decay, 0.5))
                    noise = np.multiply(amplitude, dw)


                elif self.noiseType_ == "dpd":
                    # [nBins, ]
                    dw_p = self.rng.normal(size = len(currExp))
                    dw_d = self.rng.normal(size = len(currExp))

                    amplitude_p = np.multiply (self.noiseParamsVector_[gID] , np.power(prod_rate, 0.5))
                    amplitude_d = np.multiply (self.noiseParamsVector_[gID] , np.power(decay, 0.5))

                    noise = np.multiply(amplitude_p, dw_p) + np.multiply(amplitude_d, dw_d)

                else:
                    raise KeyError(f"Unknown noise type {self.noiseType_}")

                # [nBins,] change in expression per bin
                dxdt = self.dt_ * (prod_rate - decay) + np.power(self.dt_, 0.5) * noise

                # update expression for each bin
                delIndices = []
                for bIDX, gObj in enumerate(bin_list):

                    # append new concentration level to list of expressions in bin
                    binID = gObj.binID
                    gObj.append_Conc(gObj.Conc[-1] + dxdt[bIDX])
                    gObj.incrementStep()

                    # check whether we collected enough samples
                    if len(gObj.Conc) == nReqSteps:
                        # if so, extract and save expressions at preset time snapshots
                        gObj.set_scExpression(self.scIndices_)
                        self.meanExpression [gID, binID] = np.mean(gObj.scExpression)
                        self.level2verts_[level][gIDX][binID] = gObj
                        delIndices.append(bIDX)

                # remove bins to be simulated when they are done
                sim_set[gi] = [i for j, i in enumerate(bin_list) if j not in delIndices]

                if sim_set[gi] == []:
                    delIndicesGenes.append(gi)

            # remove genes to be simulated if done
            sim_set = [i for j, i in enumerate(sim_set) if j not in delIndicesGenes]


    def simulate(self):
        for level in range(self.maxLevels_, -1, -1):
            self.CLE_simulator_(level)


    def getExpressions(self):
        ret = np.zeros((self.nBins_, self.nGenes_, self.nSC_))
        for l in range(self.maxLevels_ + 1):
            currGeneBins = self.level2verts_[l]
            for g in currGeneBins:
                gIdx = g[0].ID

                for gb in g:
                    ret[gb.binID, gIdx, :] = gb.scExpression

        return ret


def simulate(seed, config, verbose=False):

    # verbose = True
    n_vars = config["n_vars"]
    rng = np.random.default_rng(seed)
    number_sc = math.ceil(config["n_samples"] / config["n_unique_mr"])

    # sample acyclic graph with `edges_per_var` edges
    if config["graph"] == "erdos_renyi_acyclic":
        g = np.array(erdos_renyi(random.PRNGKey(seed), d=n_vars, edges_per_var=config["edges_per_var"], acyclic=True))

    elif config["graph"] == "scale_free_acyclic":
        g = np.array(scale_free(random.PRNGKey(seed), d=n_vars, power=1.0, edges_per_var=config["edges_per_var"], acyclic=True))

    elif config["graph"] == "sbm_acyclic":
        g = np.array(sbm(random.PRNGKey(seed), d=n_vars, intra_edges_per_var=config["edges_per_var"], n_blocks=5, damp=0.1, acyclic=True))

    else:
        raise ValueError(f"Unknown random graph structure model: {config['graph']}")

    # sample interaction terms K
    k = np.abs(rng.uniform(config["k_min"], config["k_max"], size=(n_vars, n_vars)))
    effect_sgn = rng.binomial(1, 0.5, size=(n_vars, n_vars)) * 2.0 - 1.0

    k = k * effect_sgn.astype(np.float32)
    assert np.array_equal(k != 0, effect_sgn != 0)
    assert set(np.unique(effect_sgn)).issubset({-1.0, 0.0, 1.0})

    # mask by graph (not needed, but for visualization purposes)
    k *= g

    # master regulator basal reproduction rate
    basal_rates = rng.uniform(config["b_min"], config["b_max"], size=(n_vars, config["n_unique_mr"]))

    # hill coeff
    hills = config["hill"] * np.ones((n_vars, n_vars))

    if verbose:
        np.set_printoptions(precision=3, suppress=True)
        print(g)
        print(k * g)

    # sample cooperative interaction terms K
    if "coop" in config and config["coop"]:
        k_coop = np.abs(rng.uniform(config["k_min"], config["k_max"], size=(n_vars, n_vars, n_vars)))
        effect_sgn_coop = rng.binomial(1, 0.5, size=(n_vars, n_vars, n_vars)) * 2.0 - 1.0
        k_coop = k_coop * effect_sgn_coop.astype(np.float32)
    else:
        k_coop = None

    """Sample targets for experiments performed (wild-type, knockout) """

    # stack a few permutations to ensure that we prioritize unseen perturbations initially, when sampling interventions
    interv_nodes_ordering = np.concatenate([rng.permutation(config["n_vars"]) for _ in range(3)])

    ko_targets = [(True, np.zeros(n_vars).astype(bool), np.ones(n_vars))]

    n_intervened = 0
    for is_train, n_interv in [(True,  config["n_intv_train"]),
                               (False, config["n_intv_test"])]:

        for j in interv_nodes_ordering[n_intervened:n_intervened + n_interv]:
            # select intervened node
            targets = np.eye(n_vars)[j].astype(bool)

            # sample strength of knockdown
            knockdown_scalars = rng.uniform(config["knockdown_min"], config["knockdown_max"], size=(n_vars,))
            knockdown_scalars = np.where(targets, knockdown_scalars, 1)

            ko_targets.append((is_train, targets, knockdown_scalars))

        n_intervened += n_interv

    # simulate for wild type and each ko target
    is_trains = []
    data = []
    intv = []
    intv_strength = []

    reference_meanExpression = None

    for ctr, (is_train, ko_target, ko_multiplier) in enumerate(ko_targets):

        if verbose:
            print(f"{ctr+1}/{len(ko_targets)}:  {ko_multiplier}")

        # setup simulator
        sim = Sergio(
            rng=rng,
            number_genes=n_vars,
            number_bins=config["n_unique_mr"],
            number_sc=number_sc,
            noise_params=config["noise_params"],
            noise_type=config["noise_type"],
            decays=config["decays"],
            sampling_state=config["sampling_state"],
            knockout_target=ko_target,
            knockout_multiplier=ko_multiplier,
            reference_meanExpression=reference_meanExpression,
            dt=config["dt"],
            safety_steps=100,
            k_coop=k_coop,
        )

        sim.custom_graph(
            g=g,
            k=k,
            b=basal_rates,
            hill=hills,
        )

        # run steady-state simulations
        assert number_sc >= 1, f"Need to have number_sc >= 1: number_sc {number_sc}, n_samples: {config['n_samples']}"
        sim.simulate()

        # Get the clean simulated expression matrix after steady_state simulations
        # shape: [number_bins (#cell types), number_genes, number_sc (#cells per type)]
        expr = sim.getExpressions()

        # Aggregate by concatenating gene expressions of all cell types into 2d array
        # [number_genes (#genes), number_bins * number_sc (#cells  = #n_unique_mr * #cells_per_type)]
        expr_agg = np.concatenate(expr, axis=1)

        # Now each row represents a gene and each column represents a simulated single-cell
        # Gene IDs match their row in this expression matrix
        # [number_bins * number_sc, number_genes]
        x = expr_agg.T
        x = rng.permutation(x, axis=0)

        # store reference gene expressions for half response fixing in interventional settings
        if reference_meanExpression is None:
            assert np.allclose(ko_target, 0), "reference_meanExpression should be set during observational phase"
            reference_meanExpression = copy.deepcopy(sim.meanExpression)
        else:
            assert not np.allclose(ko_target, 0), "reference_meanExpression should have been set before doing interventions"

        # advance rng outside for faithfullness/freshness of data in for loop
        rng = copy.deepcopy(sim.rng)

        is_trains.append(is_train)
        data.append(x)
        intv.append(ko_target)
        intv_strength.append(ko_multiplier)


    # sort into train and test and convert to data format
    dataset_fields = defaultdict(lambda: defaultdict(list))
    for is_train, x, intv, intv_mults in zip(is_trains, data, intv, intv_strength):
        dataset_fields[is_train]["data"].append(x[:config["n_samples"]])
        dataset_fields[is_train]["intv"].append(intv.astype(int))
        dataset_fields[is_train]["true_param"].append(k.T)

    for key, ddict in dataset_fields.items():
        for kkey, v in ddict.items():
            if kkey != "data": # as for real data where number of observations might be different
                dataset_fields[key][kkey] = np.stack(v)

    if verbose:
        print(f"Finished.")

    return Data(**dataset_fields[True]), Data(**dataset_fields[False])


if __name__ == "__main__":

    _ = simulate(0, dict(
        graph="erdos_renyi_acyclic",
        edges_per_var=3,
        n_vars=10,
        n_unique_mr=3,
        n_samples=100,
        n_intv_train=2,
        n_intv_test=2,
        knockdown_min=0.1,
        knockdown_max=0.9,
        b_min=1.0,
        b_max=4.0,
        k_min=3,
        k_max=10,
        hill=2.0,
        dt=0.01,
        noise_type="dpd",
        noise_params=0.5,
        decays=0.8,
        sampling_state=50,
        coop=True,
    ), verbose=False)

    exit()
