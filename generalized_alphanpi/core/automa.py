from generalized_alphanpi.utils import import_dyn_class
from generalized_alphanpi.visualize.get_automa import VisualizeAutoma

from argparse import ArgumentParser
import torch
import yaml

import numpy as np
import random

from tqdm import tqdm

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="Path to the model we want to visualize.")
    parser.add_argument("task", type=str, help="Task we want to execute")
    parser.add_argument("--config", type=str, help="Path to the file with the experiment configuration")
    parser.add_argument("--failure", action="store_true", default=False, help="Visualize an example of a failed track")
    parser.add_argument("--save-dot", action="store_true", default=False, help="Safe automa to disk as figure")
    parser.add_argument("--save-automa", action="store_true", default=False, help="Save automa model to disk as figure")
    parser.add_argument("--automa-model-path", type=str, default="./automa.pth", help="Path to the deterministic automa")
    parser.add_argument("--dot-path", type=str, default="./automa.dot", help="Path to the deterministic automa dot file")
    parser.add_argument("--max-tries", type=int, default=50, help="How many example to try")
    parser.add_argument("--seed", type=int, default=2021, help="Seed used to initialize t-sne")
    parser.add_argument("--single-core", default=True, action="store_false", help="Run everything with a single core.")
    parser.add_argument("--tree", default=False, action="store_true", help="Replace solver with decision tree")

    args = parser.parse_args()
    config = yaml.load(open(args.config),Loader=yaml.FullLoader)

    if not args.single_core:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        comm = None
        size = 1

    task_index = None
    policy = None
    buffer = None
    mcts = None
    trainer = None
    env = None
    scheduler = None
    writer = None
    automata = None

    random.seed(args.seed+rank)
    np.random.seed(args.seed+rank)
    torch.manual_seed(args.seed+rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if rank == 0:

        env = import_dyn_class(config.get("environment").get("name"))(
            **config.get("environment").get("configuration_parameters", {})
        )

        num_programs = env.get_num_programs()
        num_non_primary_programs = env.get_num_non_primary_programs()
        observation_dim = env.get_obs_dimension()
        programs_library = env.programs_library

        idx_tasks = [prog['index'] for key, prog in env.programs_library.items() if prog['level'] > 0]

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

        policy.load_state_dict(torch.load(args.model))

        MCTS_CLASS = import_dyn_class(config.get("training").get("mcts").get("name"))

        idx = env.prog_to_idx[args.task]

        mcts = MCTS_CLASS(
            env, policy, idx,
            **config.get("training").get("mcts").get("configuration_parameters")
        )
        mcts.exploration = False
        mcts.env.validation = True

        automata = VisualizeAutoma(env, operation=args.task, seed=args.seed)

    for _ in tqdm(range(0, args.max_tries//size)):

        if not args.single_core:
            mcts = comm.bcast(mcts, root=0)

        traces = mcts.sample_execution_trace()

        if not args.single_core:
            traces = comm.gather(traces, root=0)
        else:
            traces = [traces]

        if rank==0:
            for trace in traces:
                if (trace[0].rewards[0] > 0 and not args.failure) or (args.failure and trace[0].rewards[0] < 0):
                    automata.add(trace[1])

    if rank == 0:
        automata.compute(env.parsed_columns, tree=args.tree, dot_file_name=args.dot_path, save=args.save_dot)

        # Save the static automa
        if args.save_automa:
            with open(args.automa_model_path, "wb") as f:
                import dill as pickle
                pickle.dump(automata.automa, f)