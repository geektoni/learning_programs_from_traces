import torch
import pandas as pd

import numpy as np


class VisualizeAutoma:

    def __init__(self, env, operation="INTERVENE", seed=2021):
        self.env = env
        self.real_program_counts = []
        self.observations = []
        self.points = []
        self.operations = []
        self.args = []
        self.operation = operation
        self.seed = seed

        self.graph = {}
        self.graph_rules = {}
        self.automa = {}
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
                self.env.parse_observation(node.env_state)
            )

            self.points.append(
                node.h_lstm.flatten().numpy()
            )

    def compute(self, columns):
        print("[*] Compute rules given graph")

        for p in range(0, len(self.points) - 1):

            ops = (f"{self.operations[p]}({self.args[p]})", f"{self.operations[p + 1]}({self.args[p + 1]})")

            # Get current state
            state = self.operations[p]

            if not state in self.graph:
                self.graph[state] = {"arcs": {}, "data": []}
                self.envs_seen[state] = {}

            if self.real_program_counts[p] < self.real_program_counts[p + 1]:

                self.graph[state]["data"].append(self.observations[p] + [str(ops[1])])

                # Get next state
                next_state = self.operations[p + 1]

                if not next_state in self.graph.get(state):
                    self.graph.get(state)["arcs"] = {ops[1]: set()}
                else:
                    self.graph.get(state)["arcs"][ops[1]] = set()

        for k, v in self.graph.items():

            df = pd.DataFrame(v["data"], columns=columns + ["operation"], dtype=object)
            df.to_csv(f"{k}.csv", index=None)

            # Stop will be empty, so we do not process
            if k == "STOP":
                continue

            if len(df["operation"].unique()) > 1:
                print(f"[*] Getting rules for node {k}")
                self._get_rules(f"{k}.csv", k)
            else:
                print(f"[*] Add single rule for node {k}")
                self.graph[k]["arcs"][df["operation"].unique()[0]] = {'True'}
                # If there is only an operation available, then we add as a solution true
                if k not in self.automa:
                    self.automa[k] = {df["operation"].unique()[0]: [[lambda x: True]]}
                else:
                    self.automa[k][df["operation"].unique()[0]].append([lambda x: True])

        self._convert_to_dot()

    def _get_rules(self, filename, node_name):

        from minds.data import Data
        from minds.check import ConsistencyChecker
        from minds.options import Options
        from minds.mxsatsp import MaxSATSparse
        from minds.twostage import TwoStageApproach

        # setting the necessary parameters
        options = Options(["exec", "-a", "2stage", "-s", "glucose3", filename])

        # reading data from a CSV file
        data = Data(filename=options.files[0], mapfile=options.mapfile,
                    separator=options.separator, ranges=options.ranges)

        # data may be inconsistent/contradictory
        checker = ConsistencyChecker(data, options)
        if checker.status and checker.do() == False:
            print("[*] Remove inconsistencies")
            # if we do not remove inconsistency, our approach will fail
            checker.remove_inconsistent()

        # creating and calling the solver
        ruler = TwoStageApproach(data, options)
        #ruler = MaxSATSparse(data, options)
        covers = ruler.compute()

        # printing the result rules for every label/class to stdout
        for label in covers:
            for rule in covers[label]:

                body, operation = self._parse_rule(str(rule))

                rule_lambdas = self._convert_rule_into_lambda(str(rule))

                if operation in self.graph[node_name]["arcs"]:
                    self.graph[node_name]["arcs"][operation].add(body)
                else:
                    self.graph[node_name]["arcs"][operation] = {body}

                # Create deterministic automaton
                if not node_name in self.automa:
                    self.automa[node_name] = {operation: [rule_lambdas]}
                else:
                    if operation in self.automa[node_name]:
                        self.automa[node_name][operation].append(rule_lambdas)
                    else:
                        self.automa[node_name][operation] = [rule_lambdas]

    def _parse_rule(self, rule):

        body, operation = rule.replace("'", "").replace(": ", "=").split("=>")
        body = " \\n ".join([k.strip() for k in body.split(",")])
        operation = operation.replace("operation=", "").strip()

        return body, operation

    def _convert_bool(self, value):
        if value in ["True", "False"]:
            return value == "True"
        else:
            return value

    def _convert_rule_into_lambda(self, rule):

        body, operation = rule.replace("'", "").split("=>")

        rule_set = []

        rules = body.split(",")
        for r in rules:
            negation = "not" in r
            value = r.split(":")[1].strip()
            feature = r.split(":")[0].replace("not", "").strip()

            value = self._convert_bool(value)

            if negation:
                rule_set.append(lambda x: not (x[feature] == value))
            else:
                rule_set.append(lambda x: x[feature] == value)

        return rule_set

    def _convert_to_dot(self, font_size=12, color="black"):

        self.file = open("test.dot", 'w')
        self.file.write('digraph g{ \n')

        for node, childs in self.graph.items():

            self.file.write("\t" + str(node) + '\n')

            for child, rules in childs["arcs"].items():

                parsed_rules = " \\n ".join([f"({r})" for r in rules])

                child_name = child.split("(")[0]

                node_rule_name = "\t" + str(child.replace("(", "_").replace(")", "_")) + "_" + str(node) + "\t"

                action_rules = node_rule_name
                action_rules += '[ shape=box,'
                if color is not None:
                    action_rules += 'color={}, '.format(color)
                action_rules += 'label=\"{}\"'.format(
                    parsed_rules + "\\n " + child)
                action_rules += '];'

                self.file.write("\t" + action_rules + '\n')

                # Print edge
                res = '{} -> {}'.format(node_rule_name, str(child_name))
                self.file.write("\t" + res + '\n')
                res = '{} -> {} '.format(str(node), node_rule_name)
                self.file.write("\t" + res + '\n')

        self.file.write('}')
        self.file.close()
