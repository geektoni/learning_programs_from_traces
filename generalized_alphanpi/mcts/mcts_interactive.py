from generalized_alphanpi.mcts.mcts_multiple_args import MCTSMultipleArgs

import numpy as np

class MCTSInteractive(MCTSMultipleArgs):

    def __init__(self, environment, model, task_index: int, number_of_simulations: int = 100, exploration=True,
                 dir_epsilon: float = 0.03, dir_noise: float = 0.3, level_closeness_coeff: float = 3.0,
                 level_0_penalty: float = 1, qvalue_temperature: float = 1.0, temperature: float = 1.3,
                 c_puct: float = 0.5, gamma: float = 0.97, action_cost_coeff: float = 1.0,
                 action_duplicate_cost: float = 1.0, mask_actions_probability: float = 0.1):

        self.mask_actions_probability = mask_actions_probability

        super().__init__(environment, model, task_index, number_of_simulations, exploration, dir_epsilon, dir_noise,
                         level_closeness_coeff, level_0_penalty, qvalue_temperature, temperature, c_puct, gamma,
                         action_cost_coeff, action_duplicate_cost)

    def _expand_node(self, node):

        node, value, new_h, new_c, _ = super()._expand_node(node)

        # Keep only the nodes by looking at the probability
        new_nodes = [c for c in node.childs if np.random.random() >= self.mask_actions_probability]
        node.childs = new_nodes

        return node, value, new_h.clone(), new_c.clone(), len(new_nodes)


