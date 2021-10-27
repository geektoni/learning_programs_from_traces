from core.mcts_exact import MCTSExact
from agents.standard_policy import StandardPolicy
from environments.mock_env import MockEnv, MockEnvEncoder
from core.buffer.trace_buffer import PrioritizedReplayBuffer
from trainer.trainer import Trainer
from trainer.curriculum import CurriculumScheduler

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

        trainer = Trainer(policy, buffer)
        scheduler = CurriculumScheduler(0.97, num_non_primary_programs, programs_library,
                                                   moving_average=0.99)
        mcts = MCTSExact(env, policy, 1)
    else:
        policy = None
        buffer = None
        mcts = None
        trainer = None
        env = None
        scheduler = None

    for iteration in range(100):

        if rank==0:
            task_index = scheduler.get_next_task_index()
            mcts = MCTSExact(env, policy, task_index)

        for episode in range(10):

            mcts = comm.bcast(mcts, root=0)

            traces = mcts.sample_execution_trace()

            traces = comm.gather(traces, root=0)

            if rank == 0:

                act_loss, crit_loss, args_loss = trainer.train_one_step(traces)

                print(f"Done {episode}: {len(traces)}: (Buffer size {buffer.get_memory_length()})")

        if rank == 0:

            for idx in scheduler.get_tasks_of_maximum_level():
                task_level = env.get_program_level_from_index(idx)
                env = MockEnv()

                validation_rewards = trainer.perform_validation_step(env, idx)
                scheduler.update_statistics(idx, validation_rewards)
                scheduler.print_statistics()
                print('')
                print('')




            print(f"[**] Done with iteration {iteration}")


