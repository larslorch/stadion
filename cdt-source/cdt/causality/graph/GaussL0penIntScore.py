"""GIES algorithm.

.. MIT License
..
.. Copyright (c) 2018 Diviyan Kalainathan
..
.. Permission is hereby granted, free of charge, to any person obtaining a copy
.. of this software and associated documentation files (the "Software"), to deal
.. in the Software without restriction, including without limitation the rights
.. to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
.. copies of the Software, and to permit persons to whom the Software is
.. furnished to do so, subject to the following conditions:
..
.. The above copyright notice and this permission notice shall be included in all
.. copies or substantial portions of the Software.
..
.. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
.. IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
.. FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
.. AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
.. LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
.. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
.. SOFTWARE.
"""
import os
import uuid
import warnings
import networkx as nx
from pathlib import Path
from shutil import rmtree
from tempfile import gettempdir
from .model import GraphModel
from pandas import DataFrame, read_csv
from ...utils.R import RPackages, launch_R_script
from ...utils.Settings import SETTINGS


def message_warning(msg, *a, **kwargs):
    """Ignore everything except the message."""
    return str(msg) + '\n'


warnings.formatwarning = message_warning


class GaussL0penIntScore(GraphModel):

    def __init__(self, seed, verbose=False):
        """Init the model and its available arguments."""
        if not RPackages.pcalg:
            raise ImportError("R Package pcalg is not available.")

        super(GaussL0penIntScore, self).__init__()

        self.arguments = {'{SEED}': str(seed),
                          '{FOLDER}': '/tmp/cdt_GaussL0penIntScore/',
                          '{FILE}': os.sep + 'data.csv',
                          '{FILE_FLAT_TARGETS}': os.sep + 'flat_targets.csv',
                          '{FILE_FLAT_TARGET_LENGTHS}': os.sep + 'flat_target_lengths.csv',
                          '{FILE_TARGET_INDICES}': os.sep + 'target_indices.csv',
                          '{SKELETON}': 'FALSE',
                          '{GAPS}': os.sep + 'fixedgaps.csv',
                          '{SCORE}': 'GaussL0penIntScore',
                          '{VERBOSE}': 'FALSE',
                          '{OUTPUT}': os.sep + 'result.csv'}
        self.verbose = SETTINGS.get_default(verbose=verbose)


    def _run_method(self, data, flat_targets=None, flat_target_lengths=None, target_indices=None, fixedGaps=None,
                  verbose=True):
        """Setting up and running GIES with all arguments."""
        # Run GaussL0penIntScore
        self.arguments['{FOLDER}'] = Path('{0!s}/cdt_GaussL0penIntScore_{1!s}/'.format(gettempdir(), uuid.uuid4()))
        run_dir = self.arguments['{FOLDER}']
        os.makedirs(run_dir, exist_ok=True)

        assert flat_targets is not None and flat_target_lengths is not None, \
            "GIES requires targets of interventional data (list corresponding to environments)"
        assert target_indices is not None, "GIES requires indices in `targets` for each sample"
        assert not flat_targets.isnull().values.any(), "Do not pass NaNs/None in `targets`; encode observational by `-1`"
        assert not target_indices.isnull().values.any(), \
            "Do not pass NaNs/None in `target_indices`; " \
            "encode observational by mapping to an entry of `targets` with `-1`"

        def retrieve_result():
            return read_csv(Path('{}/result.csv'.format(run_dir)), delimiter=',').values

        try:
            data.to_csv(Path('{}/data.csv'.format(run_dir)), header=False, index=False)
            flat_targets.to_csv(Path('{}/flat_targets.csv'.format(run_dir)), header=False, index=False)
            flat_target_lengths.to_csv(Path('{}/flat_target_lengths.csv'.format(run_dir)), header=False, index=False)
            target_indices.to_csv(Path('{}/target_indices.csv'.format(run_dir)), header=False, index=False)

            if fixedGaps is not None:
                fixedGaps.to_csv(Path('{}/fixedgaps.csv'.format(run_dir)), index=False, header=False)
                self.arguments['{SKELETON}'] = 'TRUE'
            else:
                self.arguments['{SKELETON}'] = 'FALSE'


            method_result = launch_R_script(Path("{}/R_templates/GaussL0penIntScore.R".format(os.path.dirname(os.path.realpath(__file__)))),
                                          self.arguments, output_function=retrieve_result, verbose=verbose)
        # Cleanup
        except Exception as e:
            rmtree(run_dir)
            raise e
        except KeyboardInterrupt:
            rmtree(run_dir)
            raise KeyboardInterrupt
        rmtree(run_dir)
        return method_result


    def predict(self, df_data, graph, **kwargs):
        """Orient a graph using the method defined by the arguments.

        Depending on the type of `graph`, this function process to execute
        different functions:

        1. If ``graph`` is a ``networkx.DiGraph``, then ``self.orient_directed_graph`` is executed.
        2. If ``graph`` is a ``networkx.Graph``, then ``self.orient_undirected_graph`` is executed.
        3. If ``graph`` is a ``None``, then ``self.create_graph_from_data`` is executed.

        Args:
            df_data (pandas.DataFrame): DataFrame containing the observational data.
            graph (networkx.DiGraph or networkx.Graph or None): Prior knowledge on the causal graph.

        .. warning::
           Requirement : Name of the nodes in the graph must correspond to the
           name of the variables in df_data
        """
        self.arguments['{VERBOSE}'] = str(self.verbose).upper()

        fe = DataFrame(nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes()), weight=None).todense())
        fg = DataFrame(1 - fe.values)

        results = self._run_method(df_data, fixedGaps=fg, verbose=self.verbose, **kwargs)
        return results
