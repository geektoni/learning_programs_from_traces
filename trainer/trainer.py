from core.mcts_exact import MCTSExact

class Trainer:

    def __init__(self, policy, buffer, batch_size=50, num_updates_per_episode=5, num_validation_episodes=10):

        self.policy = policy
        self.buffer = buffer
        self.batch_size = batch_size
        self.num_updates_per_episode = num_updates_per_episode
        self.num_validation_episodes = num_validation_episodes

    def perform_validation_step(self, env, task_index):

        validation_rewards = []
        for _ in range(self.num_validation_episodes):

            mcts = MCTSExact(env, self.policy, task_index)

            # Sample an execution trace with mcts using policy as a prior
            trace = mcts.sample_execution_trace()
            task_reward = trace.task_reward

            validation_rewards.append(task_reward)
        return validation_rewards

    def train_one_step(self, traces):

        actor_losses = 0
        critic_losses = 0
        arguments_losses = 0

        # Loop over the traces and save them
        for t in traces:

            if t.task_reward < 0:
                continue

            observations = t.observations
            prog_indices = t.programs_index
            lstm_states = t.lstm_states
            policy_labels = t.mcts_policies
            rewards = t.rewards
            program_args = t.program_arguments

            if t.clean_sub_execution:
                # Generates trace
                trace = list(zip(observations, prog_indices, lstm_states, policy_labels, rewards, program_args))
                # Append trace to buffer
                self.buffer.append_trace(trace)
            else:
                # TODO: better logging
                print("Trace has not been stored in buffer.")

        if self.buffer.get_memory_length() > self.batch_size:
            for _ in range(self.num_updates_per_episode):
                batch = self.buffer.sample_batch(self.batch_size)
                if batch is not None:
                    actor_loss, critic_loss, arg_loss, _ = self.policy.train_on_batch(batch, False)
                    actor_losses += actor_loss
                    critic_losses += critic_loss
                    arguments_losses += arg_loss

        return actor_losses/self.num_updates_per_episode, critic_losses/self.num_updates_per_episode, \
            arguments_losses/self.num_updates_per_episode