from abc import ABC, abstractmethod

import random

class ExecutionTrace(ABC):
    def __init__(self):
        pass

    def get_id(self):
        return random.randint(1, 10)

class Node(ABC):

    def __init__(self, *initial_data, **kwargs):
        """
        https://stackoverflow.com/questions/2466191/set-attributes-from-dictionary-in-python
        :param initial_data:
        :param kwargs:
        """
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    @staticmethod
    def initialize_root(task_index, args, init_observation, env_state, h, c):
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
            "h_lstm": h.copy(),
            "c_lstm": c.copy(),
            "depth": 0,
            "selected": True
        })

class MCTS(ABC):
    """
    This class present the basic structure for a Monte Carlo Tree Search
    given the library, the possible arguments and the model.
    """

    def __init__(self, environment, policy, task_index: int, number_of_simulations: int=100) -> None:
        self.env = environment
        self.policy = policy
        self.task_index = task_index
        self.n_simulations = number_of_simulations
        pass

    @abstractmethod
    def _expand_node(self):
        pass

    @abstractmethod
    def _simulate(self):
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
