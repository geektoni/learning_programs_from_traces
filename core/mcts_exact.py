from core.mcts import MCTS, Node, ExecutionTrace
from utils import fix_policy, compute_q_value

import torch
import numpy as np
import copy

class MCTSExact(MCTS):

    def __init__(self, environment, model, task_index: int, number_of_simulations: int =100, exploration=True):
        super().__init__(environment, model, task_index, number_of_simulations, exploration)

    def _expand_node(self, node):

        program_index, observation, env_state, h, c, depth = (
            node.program_index,
            node.observation,
            node.env_state,
            node.h_lstm,
            node.c_lstm,
            node.depth
        )

        with torch.no_grad():
            mask = self.env.get_mask_over_actions(self.task_index)
            priors, value, new_args, new_h, new_c = self.policy.forward_once(observation, program_index, h, c)

            # mask actions
            priors = priors * torch.FloatTensor(mask)
            priors = torch.squeeze(priors)
            priors = priors.cpu().numpy()

            if self.exploration:
                priors = (1 - self.dir_epsilon) * priors + self.dir_epsilon * np.random.dirichlet([self.dir_noise] * priors.size)

            policy_indexes = [prog_idx for prog_idx, x in enumerate(mask) if x == 1]
            policy_probability = [priors[prog_idx] for prog_idx, x in enumerate(mask) if x == 1]

            #policy_probability = fix_policy(torch.FloatTensor(policy_probability)).numpy()

            # Current new nodes
            new_nodes = []

            # Initialize its children with its probability of being chosen
            for prog_index, prog_proba in zip(policy_indexes, policy_probability):

                # mask arguments
                mask_args = self.env.get_mask_over_args(prog_index)
                new_args_masked = new_args * torch.FloatTensor(mask_args)
                new_args_masked = torch.squeeze(new_args_masked)
                new_args_masked = new_args_masked.cpu().numpy()

                if self.exploration:
                    new_args_masked = (1 - self.dir_epsilon) * new_args_masked \
                                      + self.dir_epsilon * np.random.dirichlet([self.dir_noise] * new_args_masked.size)

                args_probability = [new_args_masked[arg_idx] for arg_idx, y in enumerate(mask_args) if y == 1]
                args_indexs = [arg_idx for arg_idx, y in enumerate(mask_args) if y == 1]

                #args_probability = fix_policy(torch.FloatTensor(args_probability)).numpy()

                for arg_index, args_proba in zip(args_indexs, args_probability):

                    new_child = Node({
                        "parent": node,
                        "childs": [],
                        "visit_count": 0.0,
                        "total_action_value": [],
                        "prior": float(prog_proba * args_proba),
                        "program_from_parent_index": prog_index,
                        "program_index": program_index,
                        "observation": observation,
                        "env_state": env_state.copy(),
                        "h_lstm": new_h.clone(),
                        "c_lstm": new_c.clone(),
                        "selected": False,
                        "args": self.env.arguments[arg_index],
                        "args_index": arg_index,
                        "depth": depth + 1,
                    })

                    # Add the new node in a temporary array
                    new_nodes.append(new_child)

            # Append the new nodes to graph
            node.childs = new_nodes

            # This reward will be propagated backwards through the tree
            value = float(value)
            return node, value, new_h.clone(), new_c.clone(), len(new_nodes)

    def _simulate(self, node):

        stop = False
        max_depth_reached = False
        max_recursion_reached = False
        has_expanded_a_node = False
        failed_simulation = False
        value = None
        program_level = self.env.get_program_level_from_index(node.program_index)

        while not stop and not max_depth_reached and not has_expanded_a_node and self.clean_sub_executions and not max_recursion_reached:

            if node.depth >= self.env.get_max_depth_from_level(program_level):
                max_depth_reached = True

            elif len(node.childs) == 0:
                _, value, state_h, state_c, new_childs_added = self._expand_node(node)

                has_expanded_a_node = True

                if new_childs_added == 0:
                    failed_simulation = True
                    break

            else:
                best_node = self._estimate_q_val(node)

                # Check this corner case. If this happened, then we
                # failed this simulation and its reward will be -1.
                if best_node is None:
                    failed_simulation = True
                    break
                else:
                    node = best_node

                program_to_call_index = node.program_from_parent_index
                program_to_call = self.env.get_program_from_index(program_to_call_index)
                arguments = node.args

                if program_to_call_index == self.env.get_stop_action_index():
                    stop = True

                elif self.env.get_program_level(program_to_call) == 0:
                    observation = self.env.act(program_to_call, arguments)
                    node.observation = observation
                    node.env_state = self.env.get_state().copy()

                else:
                    # check if call corresponds to a recursive call
                    if program_to_call_index == self.task_index:
                        self.recursive_call = True
                    # if never been done, compute new tree to execute program
                    if node.visit_count == 0.0:

                        sub_mcts_init_state = self.env.get_state()

                        # Copy sub_tree_params and increase node counts
                        copy_ = copy.deepcopy(self.sub_tree_params)

                        sub_mcts = MCTSExact(self.policy, self.env, program_to_call_index, **copy_)
                        sub_trace = sub_mcts.sample_execution_trace().dict()
                        sub_task_reward, sub_root_node, sub_total_nodes, sub_selected_nodes = sub_trace[7], sub_trace[6], sub_trace[14], sub_trace[15]

                        # check that sub tree executed correctly
                        self.clean_sub_executions &= (sub_task_reward > -1.0)
                        if not self.clean_sub_executions:
                            # TODO: add better logging
                            # print('program {} did not execute correctly'.format(program_to_call))
                            self.programs_failed_indices.append(program_to_call_index)
                            self.programs_failed_initstates.append(sub_mcts_init_state)

                        observation = self.env.get_observation()
                    else:
                        self.env.reset_to_state(node.env_state.copy())
                        observation = self.env.get_observation()

                    node.observation = observation
                    node.env_state = self.env.get_state().copy()

        return max_depth_reached, has_expanded_a_node, node, value, failed_simulation

    def _backpropagate(self):
        pass

    def _play_episode(self, root_node: Node):
        stop = False
        max_depth_reached = False
        illegal_action = False

        selected_nodes_count = 0

        while not stop and not max_depth_reached and not illegal_action and self.clean_sub_executions:

            program_level = self.env.get_program_level_from_index(root_node.program_index)
            root_node.selected = True

            if root_node.depth >= self.env.max_depth_dict[program_level]:
                max_depth_reached = True

            else:
                env_state = root_node.env_state.copy()

                # record obs, progs and lstm states only if they correspond to the current task at hand
                self.lstm_states.append((root_node.h_lstm, root_node.c_lstm))
                self.programs_index.append(root_node.program_index)
                self.observations.append(root_node.observation)
                self.previous_actions.append(root_node.program_from_parent_index)
                self.program_arguments.append(root_node.args)
                self.rewards.append(None)

                total_node_expanded_simulation = 0

                # Spend some time expanding the tree from your current root node
                for _ in range(self.number_of_simulations):
                    # run a simulation
                    self.recursive_call = False
                    simulation_max_depth_reached, has_expanded_node, node, value, failed_simulation = self._simulate(
                        root_node)
                    #total_node_expanded_simulation += total_node_expanded
                    #selected_nodes_count += total_sub_selected_nodes

                    # get reward
                    if failed_simulation:
                        value = -1.0
                    elif not simulation_max_depth_reached and not has_expanded_node:
                        # if node corresponds to end of an episode, backprogagate real reward
                        reward = self.env.get_reward()
                        if reward > 0:
                            value = self.env.get_reward() * (self.gamma ** node.depth)
                        else:
                            value = -1.0

                    elif simulation_max_depth_reached:
                        # if episode stops because the max depth allowed was reached, then reward = -1
                        value = -1.0

                    value = float(value)

                    exp_val = torch.exp(self.qvalue_temperature*torch.FloatTensor([value]))

                    # Propagate information backwards
                    while node.parent is not None:
                        node.visit_count += 1
                        node.total_action_value.append(value)

                        node.denom += exp_val
                        softmax = exp_val / node.denom
                        node.estimated_qval += softmax * torch.FloatTensor([value])

                        node = node.parent

                    # Root node is not included in the while loop
                    self.root_node.total_action_value.append(value)
                    self.root_node.visit_count += 1

                    self.root_node.denom += exp_val
                    softmax = exp_val / self.root_node.denom
                    self.root_node.estimated_qval += softmax * torch.FloatTensor([value])

                    # Go back to current env state
                    self.env.reset_to_state(env_state.copy())

                # Record how many nodes we expanded for this simulation
                #self.total_node_expanded[self.task_index].append(total_node_expanded_simulation)

                # Sample next action
                mcts_policy, args_policy, program_to_call_index, args_to_call_index = self._sample_policy(root_node)
                if program_to_call_index == self.task_index:
                    self.global_recursive_call = True

                # Set new root node
                root_node = [child for child in root_node.childs
                                 if child.program_from_parent_index == program_to_call_index
                                 and child.args_index == args_to_call_index]

                #print(mcts_policy, program_to_call_index, self.env.programs_library["STOP"]['index'], [child.program_from_parent_index for child in root_node.childs])

                # If we choose an illegal action from this point, we exit
                if len(root_node) == 0:
                    root_node = None
                    illegal_action = True
                else:
                    root_node = root_node[0]

                selected_nodes_count += 1

                # Record mcts policy
                self.mcts_policies.append(torch.cat([mcts_policy, args_policy], dim=1))

                # Apply chosen action
                if not illegal_action:
                    if program_to_call_index == self.env.programs_library["STOP"]['index']:
                        stop = True
                    else:
                        self.env.reset_to_state(root_node.env_state.copy())

        return root_node, max_depth_reached, illegal_action, selected_nodes_count


    def sample_execution_trace(self) -> ExecutionTrace:
        """
        Sample an execution trace from the tree by running many simulations until
        we converge or we reach the max tree depth. The execution trace is stored in
        a custom object.

        :return: an execution trace. If the reward is -1, then the execution trace is
        not valid. This means we did not reach the end of the program.
        """

        # Clear from previous content
        self.empty_previous_trace()

        init_observation, state_index  = self.env.start_task(self.task_index)
        with torch.no_grad():
            state_h, state_c = self.policy.init_tensors()
            env_init_state = self.env.get_state()

            root = Node.initialize_root(
                self.task_index, init_observation, env_init_state, state_h, state_c
            )

        self.root_node = root

        final_node, max_depth_reached, illegal_action, selected_nodes_count = self._play_episode(root)

        if not illegal_action:
            final_node.selected = True

        # compute final task reward (with gamma penalization)
        reward = self.env.get_reward()
        if reward > 0 and not illegal_action and not max_depth_reached:
            task_reward = reward * (self.gamma ** final_node.depth)
        else:
            # TODO: add logic to save failed states for retraining
            #self.programs_failed_states_indices[self.task_index].append((env_index, env_total_size))
            #self.env.update_failing_envs(self.env_init_state, self.env.get_program_from_index(self.task_index))
            task_reward = -1

        # Replace None rewards by the true final task reward
        self.rewards = list(
            map(lambda x: torch.FloatTensor([task_reward]) if x is None else torch.FloatTensor([x]), self.rewards))

        self.env.end_task()

        # Generate execution trace
        return ExecutionTrace(self.lstm_states, self.programs_index, self.observations, self.previous_actions, task_reward,
                              self.program_arguments, self.rewards, self.mcts_policies, self.clean_sub_executions)

    def _estimate_q_val(self, node):

        best_child = None
        best_val = -np.inf

        for child in node.childs:
            if child.prior > 0.0:
                q_val_action = compute_q_value(child, self.qvalue_temperature)

                action_utility = (self.c_puct * child.prior * np.sqrt(node.visit_count)
                                  * (1.0 / (1.0 + child.visit_count)))
                q_val_action += action_utility
                parent_prog_lvl = self.env.programs_library[self.env.idx_to_prog[node.program_index]]['level']
                action_prog_lvl = self.env.programs_library[self.env.idx_to_prog[child.program_from_parent_index]][
                    'level']

                if parent_prog_lvl == action_prog_lvl:
                    # special treatment for calling the same program or a level 0 action.
                    action_level_closeness = self.level_closeness_coeff * np.exp(-1)
                elif action_prog_lvl == 0:
                    action_level_closeness = self.level_closeness_coeff * np.exp(-self.level_0_penalty)
                elif action_prog_lvl > 0:
                    action_level_closeness = self.level_closeness_coeff * np.exp(-(parent_prog_lvl - action_prog_lvl))
                else:
                    # special treatment for STOP action
                    action_level_closeness = self.level_closeness_coeff * np.exp(-1)

                q_val_action += action_level_closeness

                if q_val_action > best_val:
                    best_val = q_val_action
                    best_child = child

        return best_child

    def _sample_policy(self, root_node):
        """Sample an action from the policies and q_value distributions that were previously sampled.
                Args:
                  root_node: Node to choose the best action from. It should be the root node of the tree.
                Returns:
                  Tuple containing the sampled action and the probability distribution build normalizing visits_policy.
                """
        visits_policy = []
        for child in root_node.childs:
            if child.prior > 0.0:
                visits_policy.append([child.program_from_parent_index, child.visit_count, child.args_index])

        mcts_policy = torch.zeros(1, self.env.get_num_programs())
        args_policy = torch.zeros(1, len(self.env.arguments))

        for prog_index, visit, arg_index in visits_policy:
            mcts_policy[0, prog_index] += visit
            args_policy[0, arg_index] += visit

        mcts_policy = torch.pow(mcts_policy, self.temperature)
        mcts_policy = mcts_policy / mcts_policy.sum()

        if mcts_policy.sum() == 0.0:
            mcts_policy = torch.ones(1, self.env.get_num_programs()) / self.env.get_num_programs()

        args_policy = torch.pow(args_policy, self.temperature)
        args_policy = args_policy / args_policy.sum()

        if args_policy.sum() == 0.0:
            args_policy = torch.ones(1, len(self.env.arguments)) / len(self.env.arguments)

        args_sampled = int(torch.multinomial(args_policy, 1)[0, 0])

        return mcts_policy, args_policy, int(torch.multinomial(mcts_policy, 1)[0, 0]), args_sampled

    def print_correct_execution(self, node):
        while node.parent is not None:
            program_name = self.env.get_program_from_index(node.program_from_parent_index)
            print("{}({})".format(program_name, node.args))

            node = node.parent
        print("")