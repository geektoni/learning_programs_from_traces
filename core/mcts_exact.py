from core.mcts import MCTS, Node, ExecutionTrace

import torch

class MCTSExact(MCTS):

    def __init__(self, environment, model, task_index: int, number_of_simulations: int =100):
        super().__init__(environment, model, task_index, number_of_simulations)

    def _expand_node(self):
        pass

    def _simulate(self):
        pass

    def _backpropagate(self):
        pass

    def _play_episode(self, node: Node):
        pass

    def sample_execution_trace(self) -> ExecutionTrace:
        """
        Sample an execution trace from the tree by running many simulations until
        we converge or we reach the max tree depth. The execution trace is stored in
        a custom object.

        :return: an execution trace. If the reward is -1, then the execution trace is
        not valid. This means we did not reach the end of the program.
        """

        init_observation, args = self.env.start_task(self.task_index)
        with torch.no_grad():
            state_h, state_c = self.policy.init_tensors()
            env_init_state = self.env.get_state()

            root = Node.initialize_root(
                self.task_index, args, init_observation, env_init_state, state_h, state_c
            )

        self._play_episode(root)

        self.env.end_task()

        return ExecutionTrace()
