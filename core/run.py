from core.mcts_exact import MCTSExact
from agents.standard_policy import StandardPolicy
from environments.mock_env import MockEnv, MockEnvEncoder
from core.buffer.trace_buffer import PrioritizedReplayBuffer

if __name__ == "__main__":

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        env = MockEnv()

        num_programs = env.get_num_programs()
        num_non_primary_programs = env.get_num_non_primary_programs()
        observation_dim = env.get_obs_dimension()
        programs_library = env.programs_library

        idx_tasks = [prog['index'] for key, prog in env.programs_library.items() if prog['level'] > 0]
        buffer = PrioritizedReplayBuffer(50, idx_tasks, p1=0.2)

        encoder = MockEnvEncoder(env.get_obs_dimension())
        indices_non_primary_programs = [p['index'] for _, p in programs_library.items() if p['level'] > 0]
        policy = StandardPolicy(encoder, 50, num_programs, num_non_primary_programs, 100,
                        20, indices_non_primary_programs)

        mcts = MCTSExact(env, policy, 1)
    else:
        policy = None
        buffer = None
        mcts = None

    actor_losses = 0
    critic_losses = 0
    arguments_losses = 0

    for episode in range(10):

        mcts = comm.bcast(mcts, root=0)

        traces = mcts.sample_execution_trace()

        traces = comm.gather(traces, root=0)

        if rank == 0:

            #traces = [len(t.observations) for t in traces]

            # Loop over the traces and save them
            for t in traces:

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
                    buffer.append_trace(trace)
                else:
                    print("Trace has not been stored in buffer.")

            batch_size = 10
            num_updates_per_episode = 5

            if buffer.get_memory_length() > batch_size:
                for _ in range(num_updates_per_episode):
                    batch = buffer.sample_batch(batch_size)
                    if batch is not None:
                        actor_loss, critic_loss, arg_loss, _ = policy.train_on_batch(batch, False)
                        actor_losses += actor_loss
                        critic_losses += critic_loss
                        arguments_losses += arg_loss

                        print(actor_loss, critic_loss, arg_loss)

            print(f"Done {episode}: {len(traces)} (Buffer size {buffer.get_memory_length()})")


