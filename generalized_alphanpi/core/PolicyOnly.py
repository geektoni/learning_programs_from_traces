import torch


class PolicyOnly:

    def __init__(self, policy, env, max_depth_dict):
        self.policy = policy
        self.env = env

        self.stop_index = env.programs_library["STOP"]['index']
        self.max_depth_dict = max_depth_dict
        self.clean_sub_executions = True

    def play(self, task_index):
        programs_called = []

        task_level = self.env.get_program_level_from_index(task_index)
        max_depth = self.max_depth_dict[task_level]
        depth = 0
        wrong_program = False
        cost = []

        # Start new task and initialize LSTM
        observation, _ = self.env.start_task(task_index)
        state_h, state_c, state_h_args, state_c_args = self.policy.init_tensors()

        while self.clean_sub_executions and depth <= max_depth and not wrong_program:

            # Compute priors
            priors, value, arguments, tate_h, state_c, state_h_args, state_c_args = self.policy.forward_once(observation, task_index, state_h, state_c, state_h_args, state_c_args)

            # Choose action according to argmax over priors
            program_index = torch.argmax(priors).item()
            program_name = self.env.get_program_from_index(program_index)

            # Mask arguments and choose arguments
            arguments_mask = self.env.get_mask_over_args(program_index)
            arguments = arguments * torch.FloatTensor(arguments_mask)
            arguments_index = torch.argmax(arguments).item()
            arguments_list = self.env.complete_arguments[arguments_index]

            if not self.env.can_be_called(program_index, arguments_index):
                wrong_program = True
                continue

            programs_called.append((program_name, arguments_list))

            cost.append(self.env.get_cost(program_index, arguments_index))

            depth += 1

            # Apply action
            if program_name == "STOP":
                break
            elif self.env.programs_library[program_name]['level'] == 0:
                observation = self.env.act(program_name, arguments_list)
            else:
                r, _, _ = self.play(task_index=program_index)
                if r < 1.0:
                    self.clean_sub_executions = False
                observation = self.env.get_observation()

        # Get final reward and end task
        if depth <= max_depth and not wrong_program:
            reward = self.env.get_reward()
        else:
            reward = 0.0
        self.env.end_task()

        return reward, programs_called, cost