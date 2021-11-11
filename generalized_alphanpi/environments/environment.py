from abc import ABC, abstractmethod
import numpy as np


class Environment(ABC):

    def __init__(self, prog_to_func, prog_to_precondition, prog_to_postcondition, programs_library, arguments, max_depth_dict):
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

        self.init_env()

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
            state_index, total_size = self.reset_env()

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
    def reset_env(self):
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