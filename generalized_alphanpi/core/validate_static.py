from generalized_alphanpi.utils import import_dyn_class

import numpy as np

from argparse import ArgumentParser
import torch
import yaml

import time
import os
from tqdm import tqdm

import dill

def validation_recursive_tree(model, env, action, depth):
    if action == "STOP(0)":
        return [[True, env.memory.copy()]]
    elif depth < 0:
        return [[False, env.memory.copy()]]
    else:
        node_name = action.split("(")[0]
        actions = model.get(node_name)

        if isinstance(actions, type(lambda x:0)):
            next_op = actions(None)
        else:
            next_op = actions.predict([env.get_observation().tolist()[:-1]])[0]

        if next_op != "STOP(0)":
            action_name, args = next_op.split("(")[0], next_op.split("(")[1].replace(")", "")
            if args.isnumeric():
                args = int(args)
            env.act(action_name, args)

            return validation_recursive_tree(model, env, next_op, depth-1)
        else:
            return [[True, env.memory.copy()]]


def validation_recursive(env, action, depth, alpha=0.65):

    if action == "STOP(0)":
        return [[True, env.memory.copy()]]
    elif depth < 0:
        return [[False, env.memory.copy()]]
    else:

        observation = env.memory.copy()
        node_name = action.split("(")[0]
        actions = model.get(node_name)

        total = []

        found = False
        action_boolean = []
        for a, conditions in actions.items():
            results = []
            real_result = []

            parsed_observation = {c: v for c, v in zip(env.parsed_columns, env.parse_observation(observation))}

            for c in conditions:
                tmp = [f(parsed_observation) for f in c]
                results += tmp
                real_result.append(all(tmp))

            # Get how many booleans has it satisfied
            action_boolean.append((a, sum(results)/len(results), any(real_result)))

        action_boolean.sort(key=lambda x: x[1], reverse=True)

        for next_op, rel_bool, true_bool in action_boolean:

            if true_bool or rel_bool >= alpha:

                found = True
                previous_env = env.memory.copy()

                if next_op != "STOP(0)":
                    action_name, args = next_op.split("(")[0], next_op.split("(")[1].replace(")", "")
                    if args.isnumeric():
                        args = int(args)
                    env.act(action_name, args)

                total += validation_recursive(env, next_op, depth-1, alpha)

                env.memory = previous_env

        if not found:
            return [[False, env.memory.copy()]]
        else:
            return total


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="Path to the automa model we want to validate.")
    parser.add_argument("task", type=str, help="Task we want to execute")
    parser.add_argument("--config", type=str, help="Path to the file with the experiment configuration")
    parser.add_argument("--alpha", type=float, default=0.65, help="Percentage of successful rules satisfied")
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

    env = None
    reward = 0
    results_file = None

    if rank == 0:

        env = import_dyn_class(config.get("environment").get("name"))(
            **config.get("environment").get("configuration_parameters", {}),
            **config.get("validation").get("environment").get("configuration_parameters", {})
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

        idx = env.prog_to_idx[args.task]
        failures = 0

        ts = time.localtime(time.time())
        date_time = 'validation-static-{}-{}_{}_{}-{}_{}_{}.csv'.format(args.task, ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])

        results_filename = config.get("validation").get("save_results_name")+date_time
        results_file = open(
            os.path.join(config.get("validation").get("save_results"), results_filename), "w"
        )

        with open(args.model, "rb") as f:
            import dill as pickle
            model = pickle.load(f)

        # perform validation, not training
        env.validation = True

        reward = 0

    iterations = min(int(config.get("validation").get("iterations")), len(env.data))

    for _ in tqdm(range(0, iterations//size)):

        if not args.single_core:
            env = comm.bcast(env, root=0)

        idx = env.prog_to_idx[args.task]

        _, state_index = env.start_task(idx)

        max_depth = env.max_depth_dict.get(1)

        next_action = "INTERVENE(0)"

        if args.tree:
            results = validation_recursive_tree(model, env, next_action, max_depth)
        else:
            results = validation_recursive(env, next_action, max_depth, args.alpha)

        if not args.single_core:
            results = comm.gather(results, root=0)
        else:
            results = [results]

        if rank == 0:
            for R in results:
                for r in R:
                    env.memory = r[1]
                    if env.prog_to_postcondition[env.get_program_from_index(idx)](None, None) and r[0]:
                        reward += 1
                        break

        env.end_task()

    if rank == 0:
        print("Correct:", reward)
        print("Failures:", iterations-reward)
        results_file.write(f"correct,wrong" + '\n')
        results_file.write(f"{reward}, {iterations-reward}" + '\n')
        results_file.close()


