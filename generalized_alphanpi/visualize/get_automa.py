import torch
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

class VisualizeAutoma:

    def __init__(self, env, operation="INTERVENE", seed=2021):
        self.env = env
        self.real_program = []
        self.real_program_counts = []
        self.points = []
        self.operations = []
        self.operation = operation
        self.seed = 2021

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
            self.points.append(
                (node.h_lstm.flatten()+node.h_lstm_args.flatten()).numpy()
                #encoder(torch.FloatTensor(node.observation)).numpy()
            )

    def compute(self):
        print("[*] Executing TSNE")
        self.reduced_points = TSNE(n_components=2,
                                   init="pca",
                                   perplexity=20,
                                   method="exact",
                                   n_jobs=-1,
                                   random_state=self.seed,
                                   ).fit_transform(self.points)
        #self.reduced_points = PCA(n_components=2).fit_transform(self.points)
        self.reduced_points = pd.DataFrame(self.reduced_points, columns=["x", "y"])
        self.reduced_points["operations"] = self.operations

    def _plot_lines(self, fig):

        op_centroids = {}

        for op in self.reduced_points["operations"].unique():
            x = self.reduced_points[self.reduced_points.operations == op]["x"].mean()
            y = self.reduced_points[self.reduced_points.operations == op]["y"].mean()

            op_centroids[op] = (x, y)

        arrows = set()

        for p in range(0, len(self.reduced_points)-1):

            ops = (self.operations[p], self.operations[p+1])
            if ops in arrows:
                continue
            else:
                arrows.add(ops)

            begin = op_centroids.get(ops[0])
            end = op_centroids.get(ops[1])

            if self.real_program_counts[p] < self.real_program_counts[p+1]:
                fig.arrow(begin[0], begin[1],
                          end[0]-begin[0],
                          end[1]-begin[1],
                          head_width=2, length_includes_head=True)


    def plot(self):
        print("[*] Plot values")
        print(self.real_program, self.real_program_counts)
        print(self.reduced_points)
        fig = sns.scatterplot(x="x", y="y", hue="operations", data=self.reduced_points)
        self._plot_lines(fig)
        plt.show()
