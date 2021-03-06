from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import pickle
import torch


class Environment(ABC):

    def __init__(self, prog_to_func, prog_to_precondition, prog_to_postcondition, programs_library, arguments,
                 max_depth_dict, prog_to_cost=None, complete_arguments=None, sample_from_errors_prob=0.3,
                 custom_tensorboard_metrics=None, validation=False):

        self.prog_to_func = prog_to_func
        self.prog_to_precondition = prog_to_precondition
        self.prog_to_postcondition = prog_to_postcondition
        self.programs_library = programs_library

        self.programs = list(self.programs_library.keys())
        self.primary_actions = [prog for prog in self.programs_library if self.programs_library[prog]['level'] <= 0]
        self.mask = dict(
            (p, self._get_available_actions(p)) for p in self.programs_library if self.programs_library[p]["level"] > 0)

        self.prog_to_idx = dict((prog, elems["index"]) for prog, elems in self.programs_library.items())
        self.idx_to_prog = dict((idx, prog) for (prog, idx) in self.prog_to_idx.items())

        self.maximum_level = max([x['level'] for prog, x in self.programs_library.items()])

        self.has_been_reset = True

        self.max_depth_dict = max_depth_dict

        self.tasks_dict = {}
        self.tasks_list = []

        self.arguments = arguments
        self.complete_arguments = complete_arguments

        self.prog_to_cost = prog_to_cost

        self.failed_execution_envs = {
            k: [] for k in self.programs_library
        }
        self.max_failed_envs = 200
        self.sample_from_errors_prob = sample_from_errors_prob
        self.validation = validation

        if custom_tensorboard_metrics is None:
            custom_tensorboard_metrics = {}
        self.custom_tensorboard_metrics = custom_tensorboard_metrics

        self.init_env()

    def setup_dataset(self, dataset, bad_class_value, target_column, predicted_column):

        self.data = pd.read_csv(dataset, sep=",")
        self.data = self.data.dropna()  # Drop columns with na
        self.data = self.data[self.data[target_column] == bad_class_value]
        self.data = self.data[self.data[predicted_column] == self.data[target_column]]

        self.y = self.data[target_column]
        self.y.reset_index(drop=True, inplace=True)

        self.data = self.data.drop(columns=[target_column, predicted_column])
        self.data.reset_index(drop=True, inplace=True)

    def setup_system(self, boolean_cols, categorical_cols, encoder, scaler,
                      classifier, net_class, sample_env, net_layers=5, net_size=108):
        self.parsed_columns = boolean_cols + categorical_cols

        self.complete_arguments = []

        for k, v in self.arguments.items():
            self.complete_arguments += v

        self.arguments_index = [(i, v) for i, v in enumerate(self.complete_arguments)]

        self.max_depth_dict = {1: 5}

        for idx, key in enumerate(sorted(list(self.programs_library.keys()))):
            self.programs_library[key]['index'] = idx

        # Load encoder
        self.data_encoder = pickle.load(open(encoder, "rb"))
        self.data_scaler = pickle.load(open(scaler, "rb"))

        # Load the classifier
        checkpoint = torch.load(classifier)
        self.classifier = net_class(net_size, layers=net_layers)  # Taken empirically from the classifier
        self.classifier.load_state_dict(checkpoint)

        # Needed for validation
        self.sample_env = sample_env
        self.current_idx = 0

        # Custom metric we want to print at each iteration
        self.custom_tensorboard_metrics = {
            "call_to_the_classifier": 0
        }


    def start_task(self, task_index):

        task_name = self.get_program_from_index(task_index)
        assert self.prog_to_precondition[task_name], 'cant start task {} ' \
                                                     'because its precondition is not verified'.format(task_name)
        self.current_task_index = task_index
        self.tasks_list.append(task_index)

        state_index = -1
        total_size = -1

        if len(self.tasks_dict.keys()) == 0:
            # reset env
            state_index, total_size = self.reset_env(task_index)

        # store init state
        init_state = self.get_state()
        self.tasks_dict[len(self.tasks_list)] = init_state

        return self.get_observation(), state_index

    @abstractmethod
    def get_observation(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def reset_env(self, task_index):
        pass

    @abstractmethod
    def init_env(self):
        pass

    @abstractmethod
    def get_obs_dimension(self):
        pass

    @abstractmethod
    def reset_to_state(self, state):
        pass

    def get_num_non_primary_programs(self):
        return len(self.programs) - len(self.primary_actions)

    def get_num_programs(self):
        return len(self.programs)

    def end_task(self):
        """
        Ends the last tasks that has been started.
        """
        del self.tasks_dict[len(self.tasks_list)]
        self.tasks_list.pop()
        if self.tasks_list:
            self.current_task_index = self.tasks_list[-1]
        else:
            self.current_task_index = None
            self.has_been_reset = False

    def get_max_depth_from_level(self, level):

        if level in self.max_depth_dict:
            return self.max_depth_dict[level]
        else:
            raise ValueError(f"Level {level} is not present in {self.max_depth_dict}")

    def _get_available_actions(self, program):
        level_prog = self.programs_library[program]["level"]
        assert level_prog > 0
        mask = np.zeros(len(self.programs))
        for prog, elems in self.programs_library.items():
            if elems["level"] < level_prog:
                mask[elems["index"]] = 1
        return mask

    def get_program_from_index(self, program_index):
        """Returns the program name from its index.
        Args:
          program_index: index of desired program
        Returns:
          the program name corresponding to program index
        """
        return self.idx_to_prog[program_index]

    def get_program_level(self, program):
        return self.programs_library[program]['level']

    def get_program_level_from_index(self, program_index):
        """
        Args:
            program_index: program index
        Returns:
            the level of the program
        """
        program = self.get_program_from_index(program_index)
        return self.programs_library[program]['level']

    def get_mask_over_actions(self, program_index):

        program = self.get_program_from_index(program_index)
        assert program in self.mask, "Error program {} provided is level 0".format(program)
        mask = self.mask[program].copy()
        # remove actions when pre-condition not satisfied
        for program, program_dict in self.programs_library.items():
            if not self.prog_to_precondition[program]():
                mask[program_dict['index']] = 0
        return mask

    def get_mask_over_args(self, program_index):
        """
        Return the available arguments which can be called by that given program
        :param program_index: the program index
        :return: a max over the available arguments
        """

        program = self.get_program_from_index(program_index)
        permitted_arguments = self.programs_library[program]["args"]
        mask = np.zeros(len(self.arguments))
        for i in range(len(self.arguments)):
            if sum(self.arguments[i]) in permitted_arguments:
                mask[i] = 1
        return mask

    def can_be_called(self, program_index, args_index):
        program = self.get_program_from_index(program_index)
        args = self.complete_arguments[args_index]

        mask_over_args = self.get_mask_over_args(program_index)
        if mask_over_args[args_index] == 0:
            return False

        return self.prog_to_precondition[program](args)

    def get_cost(self, program_index, args_index):

        if self.prog_to_cost is None:
            return 0

        program = self.get_program_from_index(program_index)
        args = self.complete_arguments[args_index]

        return self.prog_to_cost[program](args)


    def act(self, primary_action, arguments=None):
        assert self.has_been_reset, 'Need to reset the environment before acting'
        assert primary_action in self.primary_actions, 'action {} is not defined'.format(primary_action)
        self.prog_to_func[primary_action](arguments)
        return self.get_observation()

    def get_reward(self):
        task_init_state = self.tasks_dict[len(self.tasks_list)]
        state = self.get_state()
        current_task = self.get_program_from_index(self.current_task_index)
        current_task_postcondition = self.prog_to_postcondition[current_task]
        return int(current_task_postcondition(task_init_state, state))

    @abstractmethod
    def get_additional_parameters(self):
        return {}

    def get_state_str(self, state):
        return ""

    @abstractmethod
    def compare_state(self, state_a, state_b):
        pass

    def update_failing_envs(self, state, program):
        """
        Update failing env count
        :param env: current failed env
        :param program: current failed program
        :return:
        """

        if self.failed_execution_envs is None:
            raise Exception("The failed envs are None! Error sampling is not implemented")

        # Do not update if we are running in validation mode
        if len(self.failed_execution_envs[program]) == 0:
            self.failed_execution_envs[program].append((state, 1, 1000))
        else:
            found = False
            for i in range(len(self.failed_execution_envs[program])):
                if self.compare_state(state, self.failed_execution_envs[program][i][0]):
                    self.failed_execution_envs[program][i] = (
                    self.failed_execution_envs[program][i][0], self.failed_execution_envs[program][i][1] + 1,
                    self.failed_execution_envs[program][i][2] + 1)
                    found = True
                    break
                else:
                    self.failed_execution_envs[program][i] = (
                    self.failed_execution_envs[program][i][0], self.failed_execution_envs[program][i][1],
                    self.failed_execution_envs[program][i][2] - 1)
            if not found:
                # Remove the failed program with the least life from the list to make space for the new one
                if len(self.failed_execution_envs[program]) >= self.max_failed_envs:
                    self.failed_execution_envs[program].sort(key=lambda t: t[2])
                    del self.failed_execution_envs[program][0]
                self.failed_execution_envs[program].append((state, 1, 1000))

    def sample_from_failed_state(self, program):
        """
        Return the dictionary where to sample from.
        :param program: program we are resetting
        :return: the dictionary
        """
        result = None
        if np.random.random_sample() < self.sample_from_errors_prob \
                and len(self.failed_execution_envs[program]) > 0 \
                and not self.validation:
            env = self.failed_execution_envs
            total_errors = sum([x[1] for x in env[program]])
            sampling_prob = [x[1]/total_errors for x in env[program]]
            index = np.random.choice(len(env[program]), p=sampling_prob)
            result = env[program][index][0].copy()

        return result