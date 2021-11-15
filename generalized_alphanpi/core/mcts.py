from abc import ABC, abstractmethod

import numpy as np

class ExecutionTrace(ABC):

    def __init__(self, lstm_states, programs_index, observations, previous_actions, task_reward,
                              program_arguments, rewards, mcts_policies, clean_sub_execution = True):

        self.lstm_states = lstm_states
        self.programs_index = programs_index
        self.observations = observations
        self.previous_actions = previous_actions
        self.task_reward = task_reward
        self.program_arguments = program_arguments
        self.rewards = rewards
        self.mcts_policies = mcts_policies
        self.clean_sub_execution = clean_sub_execution

    def get_trace_programs(self):
        result =  [(p, arg) for p, arg in zip(self.previous_actions, self.program_arguments)]
        # Discard the first element (since it will have a None action)
        return result[1:]

    def flatten(self):
        return list(zip(self.observations,
                        self.programs_index,
                        self.lstm_states,
                        self.mcts_policies,
                        self.rewards,
                        self.program_arguments))

class Node(ABC):

    def __init__(self, *initial_data, **kwargs):
        """
        https://stackoverflow.com/questions/2466191/set-attributes-from-dictionary-in-python
        :param initial_data:
        :param kwargs:
        """

        self.parent = None
        self.childs = []
        self.visit_count = 0
        self.total_action_value = []
        self.prior = None
        self.program_index = None
        self.program_from_parent_index = None
        self.observation = None
        self.env_state = None
        self.h_lstm = None
        self.c_lstm = None
        self.depth = 0
        self.selected = False
        self.args = None
        self.args_index = None
        self.denom = 0.0
        self.estimated_qval = 0.0

        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    @staticmethod
    def initialize_root(task_index, init_observation, env_state, h, c):
        return Node({
            "parent": None,
            "childs": [],
            "visit_count": 1,
            "total_action_value": [],
            "prior": None,
            "program_index": task_index,
            "program_from_parent_index": None,
            "observation": init_observation,
            "env_state": env_state,
            "h_lstm": h.clone(),
            "c_lstm": c.clone(),
            "depth": 0,
            "selected": True,
            "args": np.array([0,0,0]),
            "args_index": 0,
            "denom": 0.0,
            "estimated_qval": 0.0
        })

    def to_dict(self):
        return {
            "parent": self.parent,
            "childs": self.childs,
            "visit_count": self.visit_count,
            "total_action_value": self.total_action_value,
            "prior": self.prior,
            "program_index": self.program_index,
            "program_from_parent_index": self.program_from_parent_index,
            "observation": self.observation,
            "env_state": self.env_state,
            "h_lstm": self.h_lstm,
            "c_lstm": self.c_lstm,
            "depth": self.depth,
            "selected": self.selected,
            "args": self.args,
            "args_index": self.args_index,
            "denom": self.denom,
            "estimated_qval": self.estimated_qval
        }

class MCTS(ABC):
    """
    This class present the basic structure for a Monte Carlo Tree Search
    given the library, the possible arguments and the model.
    """

    def __init__(self, environment, policy, task_index: int, number_of_simulations: int=100, exploration=True,
                 dir_noise: float=0.03, dir_epsilon: float=0.3,
                 level_closeness_coeff: float=3.0, level_0_penalty: float=1, qvalue_temperature: float=1.0,
                 temperature: float=1.3, c_puct: float=0.5, gamma: float=0.97, action_cost_coeff: float=1.0) -> None:
        self.env = environment
        self.policy = policy
        self.task_index = task_index
        self.number_of_simulations = number_of_simulations
        self.exploration = exploration
        self.dir_epsilon = dir_noise
        self.dir_noise = dir_epsilon

        self.clean_sub_executions = True
        self.sub_tree_params = {}
        self.level_closeness_coeff = level_closeness_coeff
        self.level_0_penalty = level_0_penalty
        self.qvalue_temperature = qvalue_temperature
        self.temperature = temperature
        self.c_puct = c_puct
        self.gamma = gamma
        self.action_cost_coeff = action_cost_coeff

        self.root_node = None

        # These list will store the failed indices
        self.programs_failed_indices = []
        self.programs_failed_initstates = []

        self.lstm_states = []
        self.programs_index = []
        self.observations = []
        self.previous_actions = []
        self.program_arguments = []
        self.rewards = []
        self.mcts_policies = []


    def empty_previous_trace(self):
        self.lstm_states = []
        self.programs_index = []
        self.observations = []
        self.previous_actions = []
        self.program_arguments = []
        self.rewards = []
        self.mcts_policies = []

    @abstractmethod
    def _expand_node(self, node: Node):
        pass

    @abstractmethod
    def _simulate(self, node: Node):
        pass

    @abstractmethod
    def _backpropagate(self):
        pass

    @abstractmethod
    def _play_episode(self, node: Node):
        pass

    @abstractmethod
    def sample_execution_trace(self):
        pass

    @abstractmethod
    def _estimate_q_val(self, node):
        pass

    @abstractmethod
    def _sample_policy(self, node):
        pass
