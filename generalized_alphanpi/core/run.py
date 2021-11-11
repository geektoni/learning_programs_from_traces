from generalized_alphanpi.core.buffer.trace_buffer import PrioritizedReplayBuffer
from generalized_alphanpi.trainer.trainer import Trainer
from generalized_alphanpi.trainer.curriculum import CurriculumScheduler
from generalized_alphanpi.utils import import_dyn_class

import torch
import numpy as np
import random

from tensorboardX import SummaryWriter

from argparse import ArgumentParser
import yaml
import time
import os

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the file with the experiment configuration")

    args = parser.parse_args()
    config = yaml.load(open(args.config),Loader=yaml.FullLoader)

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    task_index = None
    policy = None
    buffer = None
    mcts = None
    trainer = None
    env = None
    scheduler = None
    writer = None

    seed = config.get("general").get("seed", 0)

    random.seed(seed+rank)
    np.random.seed(seed+rank)
    torch.manual_seed(seed+rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    MCTS_CLASS = import_dyn_class(config.get("training").get("mcts").get("name"))

    ts = time.localtime(time.time())
    date_time = '{}_{}_{}-{}_{}_{}.model'.format(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])
    save_model_path = os.path.join(config.get("general").get("save_model_dir"),
                                   config.get("general").get("save_model_name")+"-"+date_time)

    if rank == 0:

        env = import_dyn_class(config.get("environment").get("name"))(
            **config.get("environment").get("configuration_parameters", {})
        )

        num_programs = env.get_num_programs()
        num_non_primary_programs = env.get_num_non_primary_programs()
        observation_dim = env.get_obs_dimension()
        programs_library = env.programs_library

        idx_tasks = [prog['index'] for key, prog in env.programs_library.items() if prog['level'] > 0]

        # Initialize the replay buffer. It is needed to store the various traces for training
        buffer = PrioritizedReplayBuffer(config.get("training").get("replay_buffer").get("size"),
                                         idx_tasks,
                                         p1=config.get("training").get("replay_buffer").get("sampling_correct_probability")
                                         )

        # Set up the encoder needed for the environment
        encoder = import_dyn_class(config.get("environment").get("encoder").get("name"))(
            env.get_obs_dimension(),
            config.get("environment").get("encoder").get("configuration_parameters").get("encoding_dim")
        )

        indices_non_primary_programs = [p['index'] for _, p in programs_library.items() if p['level'] > 0]

        additional_arguments_from_env = env.get_additional_parameters()

        policy = import_dyn_class(config.get("policy").get("name"))(
            encoder,
            config.get("policy").get("hidden_size"),
            num_programs, num_non_primary_programs,
            config.get("policy").get("embedding_dim"),
            config.get("policy").get("encoding_dim"),
            indices_non_primary_programs,
            **additional_arguments_from_env
        )

        # Set up the trainer algorithm
        trainer = Trainer(policy, buffer, config.get("training").get("mcts").get("name"),
                          batch_size=config.get("training").get("trainer").get("batch_size"))

        # Set up the curriculum scheduler that decides the next experiments to be done
        scheduler = CurriculumScheduler(config.get("training").get("curriculum_scheduler").get("next_action_accuracy"),
                                        num_non_primary_programs, programs_library,
                                        moving_average=config.get("training").get("curriculum_scheduler").get("moving_average"))

        mcts = import_dyn_class(config.get("training").get("mcts").get("name"))(
            env, policy, 1,
            **config.get("training").get("mcts").get("configuration_parameters")
        )

        writer = SummaryWriter(config.get("general").get("tensorboard_dir"))

    for iteration in range(config.get("training").get("num_iterations")):

        if rank==0:
            task_index = scheduler.get_next_task_index()
            mcts = MCTS_CLASS(
                env, policy, task_index,
                **config.get("training").get("mcts").get("configuration_parameters")
            )

        for episode in range(config.get("training").get("num_episodes_per_iteration")):

            mcts = comm.bcast(mcts, root=0)

            traces, _ = mcts.sample_execution_trace()

            traces = comm.gather(traces, root=0)

            if rank == 0:

                act_loss, crit_loss, args_loss = trainer.train_one_step(traces)

                v_task_name = env.get_program_from_index(task_index)
                writer.add_scalar("loss/" + v_task_name + "/actor", act_loss, iteration)
                writer.add_scalar("loss/" + v_task_name + "/value", crit_loss, iteration)
                writer.add_scalar("loss/" + v_task_name + "/arguments", args_loss, iteration)

                print(f"Done {episode}/10! s:{buffer.get_total_successful_traces()}, f:{buffer.get_total_failed_traces()}")

        if rank == 0:

            for idx in scheduler.get_tasks_of_maximum_level():
                task_level = env.get_program_level_from_index(idx)
                mcts = MCTS_CLASS(
                    env, policy, idx,
                    config.get("environment").get("configuration_parameters", {})
                )

                validation_rewards = trainer.perform_validation_step(env, idx)
                scheduler.update_statistics(idx, validation_rewards)
                scheduler.print_statistics()
                print('')
                print('')

                for idx in scheduler.get_tasks_of_maximum_level():
                    v_task_name = env.get_program_from_index(idx)
                    # record on tensorboard
                    writer.add_scalar('validation/' + v_task_name, scheduler.get_statistic(idx), iteration)

            print(f"[**] Done with iteration {iteration}")

            # Save policy
            if config.get("general").get("save_model"):
                torch.save(policy.state_dict(), save_model_path)


