from core.mcts_exact import MCTSExact
from agents.standard_policy import StandardPolicy
from environments.mock_env import MockEnv, MockEnvEncoder

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

        encoder = MockEnvEncoder(env.get_obs_dimension())
        indices_non_primary_programs = [p['index'] for _, p in programs_library.items() if p['level'] > 0]
        policy = StandardPolicy(encoder, 50, num_programs, num_non_primary_programs, 100,
                        20, indices_non_primary_programs)

        mcts = MCTSExact(env, policy, 1)
    else:
        mcts = None

    for episode in range(10):

        mcts = comm.bcast(mcts, root=0)

        traces = mcts.sample_execution_trace()

        traces = comm.gather(traces, root=0)

        if rank == 0:
            traces = [t.observations for t in traces]
            print(f"Done {episode}: {len(traces)}")


