import torch
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np

import seaborn as sns
sns.set_context("paper")
import matplotlib.pyplot as plt

from matplotlib import rcParams

# figure size in inches
rcParams['figure.figsize'] = 12,6

class VisualizeAutoma:

    def __init__(self, env, operation="INTERVENE", seed=2021, draw_arrows=False):
        self.env = env
        self.real_program = []
        self.real_program_counts = []
        self.observations = []
        self.points = []
        self.operations = []
        self.args = []
        self.operation = operation
        self.seed = seed
        self.draw_arrows = draw_arrows

        self.graph = {}
        self.envs_seen = {}

    def get_breadth_first_nodes(self, root_node):
        '''
        Performs a breadth first search inside the tree.

        Args:
            root_node: tree root node

        Returns:
            list of the tree nodes sorted by depths
        '''
        nodes = []
        stack = [root_node]
        while stack:
            cur_node = stack[0]
            stack = stack[1:]
            nodes.append(cur_node)
            for child in cur_node.childs:
                stack.append(child)
        return nodes

    def add(self, encoder, node):
        counter = 0
        nodes = self.get_breadth_first_nodes(node)

        nodes = list(filter(lambda x: x.visit_count > 0, nodes))

        for idx, tmp_node in enumerate(nodes):
            tmp_node.index = idx

        # gather nodes per depth
        max_depth = nodes[-1].depth
        nodes_per_depth = {}
        for d in range(0, max_depth + 1):
            nodes_per_depth[d] = list(filter(lambda x: x.depth == d, nodes))

        for d in range(0, max_depth + 1):
            nodes_this_depth = nodes_per_depth[d]
            for tmp_node in nodes_this_depth:
                if tmp_node.selected:
                    self.add_point(encoder, tmp_node)
                    counter += 1
                    self.real_program_counts.append(counter)


    def add_point(self, encoder, node):

        with torch.no_grad():
            if node.program_from_parent_index is None:
                self.operations.append(
                    self.operation
                )
            else:
                self.operations.append(
                    self.env.get_program_from_index(node.program_from_parent_index)
                )

            self.args.append(
                node.args
            )

            self.observations.append(
                node.env_state
            )

            self.points.append(
                (node.h_lstm.flatten()/2+node.c_lstm.flatten()/2).numpy()
                #encoder(torch.FloatTensor(node.observation)).numpy()
            )

    def _compute_min_distance(self, point, centroids):

        min_distance = np.inf
        found_centroid = None

        for k, v in centroids.items():

            tmp = np.linalg.norm(np.array(v) - np.array(point))
            if tmp < min_distance:
                min_distance = tmp
                found_centroid = k

        return found_centroid

    def _get_rules(self, filename):

        from minds.data import Data
        from minds.check import ConsistencyChecker
        from minds.options import Options
        from minds.mxsatsp import MaxSATSparse
        from minds.mxsatls import MaxSATLitsSep
        from minds.twostage import TwoStageApproach

        # setting the necessary parameters
        options = Options()
        options.solver = 'glucose3'
        options.cover = 'gurobi'
        options.opt = True
        options.verb = 0  # verbosity level

        # reading data from a CSV file
        data = Data(filename=filename, separator=',',mapfile=options.mapfile, ranges=options.ranges)

        # data may be inconsistent/contradictory
        checker = ConsistencyChecker(data, options)
        if checker.status and checker.do() == False:
            print("[*] Remove inconsistencies")
            # if we do not remove inconsistency, our approach will fail
            checker.remove_inconsistent()

        # creating and calling the solver
        #ruler = MaxSATSparse(data, options)
        #ruler = MaxSATLitsSep(data, options)
        ruler = TwoStageApproach(data, options)
        covers = ruler.compute()

        # printing the result rules for every label/class to stdout
        for label in covers:
            for rule in covers[label]:
                print('rule:', rule)


    def compute(self, columns):
        print("[*] Executing TSNE")

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.reduced_points = TSNE(n_components=2,
                                       init="pca",
                                       metric="euclidean",
                                       perplexity=20,
                                       #method="exact",
                                       n_jobs=-1,
                                       random_state=self.seed,
                                       ).fit_transform(self.points)

        self.reduced_points = pd.DataFrame(self.reduced_points, columns=["x", "y"])
        self.reduced_points["operations"] = self.operations

        operation_centroids = {}

        for op in self.reduced_points["operations"].unique():
            x = self.reduced_points[self.reduced_points.operations == op]["x"].mean()
            y = self.reduced_points[self.reduced_points.operations == op]["y"].mean()

            operation_centroids[op] = [x, y]

        for p in range(0, len(self.reduced_points)-1):

            ops = (f"{self.operations[p]}({self.args[p]})", f"{self.operations[p+1]}({self.args[p+1]})")

            # Get current state
            if self.real_program_counts[p] == 0:
                state = operation_centroids["INTERVENE"]
            else:
                x, y = self.reduced_points.values[p][0], self.reduced_points.values[p][1]
                state = self._compute_min_distance([x, y], operation_centroids)

            if not state in self.graph:
                self.graph[state] = {"arcs": {}, "data": []}
                self.envs_seen[state] = {}

            if self.real_program_counts[p] < self.real_program_counts[p+1]:

                if "_".join(map(str, self.observations[p])) in self.envs_seen[state]:
                    if self.envs_seen[state]["_".join(map(str, self.observations[p]))] != str(ops[1]):
                        print(f"{state}: ERROR")
                    else:
                        self.graph[state]["data"].append(self.observations[p] + [str(ops[1])])
                else:
                    self.envs_seen[state]["_".join(map(str, self.observations[p]))] = str(ops[1])
                    self.graph[state]["data"].append(self.observations[p] + [str(ops[1])])

                # Get next state
                x, y = self.reduced_points.values[p+1][0], self.reduced_points.values[p+1][1]
                next_state = self._compute_min_distance([x, y], operation_centroids)

                if not next_state in self.graph.get(state):
                    self.graph.get(state)["arcs"][next_state] = {ops[1]}
                else:
                    self.graph.get(state)["arcs"][next_state].add(ops[1])

        for k, v in self.graph.items():

            df = pd.DataFrame(v["data"], columns=columns+["operation"], dtype=object)

            #df["outcome"] = df["outcome"].apply(lambda x: 1 if x==0 else 0)
            df.to_csv(f"{k}.csv", index=None)

            print(f"Getting rules for node {k}")
            #self._get_rules(f"{k}.csv")
            print()


    def _plot_lines(self, fig):

        op_centroids = {}

        for op in self.reduced_points["operations"].unique():
            x = self.reduced_points[self.reduced_points.operations == op]["x"].mean()
            y = self.reduced_points[self.reduced_points.operations == op]["y"].mean()

            op_centroids[op] = (x, y)

        arrows = set()

        plot_path = True

        for p in range(0, len(self.reduced_points)-1):

            ops = (f"{self.operations[p]}({self.args[p]})", f"{self.operations[p+1]}({self.args[p+1]})")
            if ops in arrows:
                continue
            else:
                arrows.add(ops)

            begin = op_centroids.get(self.operations[p])
            end = op_centroids.get(self.operations[p+1])

            if self.real_program_counts[p] < self.real_program_counts[p+1]:
                if plot_path:
                    fig.arrow(begin[0], begin[1],
                          end[0]-begin[0],
                          end[1]-begin[1],
                          head_width=1, length_includes_head=True, width=0.09, color="r")
                    fig.text(begin[0]+(end[0]-begin[0])/2, begin[1]+(end[1]-begin[1])/2, ops[0])
                else:
                    fig.arrow(begin[0], begin[1],
                              end[0] - begin[0],
                              end[1] - begin[1],
                              head_width=1, length_includes_head=True)
            else:
                plot_path = False

    def plot(self, save=False):
        print("[*] Plot values")
        fig = sns.scatterplot(x="x", y="y", hue="operations", data=self.reduced_points)

        if self.draw_arrows:
            self._plot_lines(fig)

        if save:
            plt.tight_layout()
            plt.savefig("output_automa.png", dpi=400)
        else:
            plt.show()
