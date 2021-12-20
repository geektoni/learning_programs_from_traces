from generalized_alphanpi.utils import import_dyn_class, get_cost_from_env

import numpy as np
import pandas as pd

from argparse import ArgumentParser
import yaml

import time
import os
from tqdm import tqdm

import dill

def validation_recursive_tree(model, env, action, depth, cost, action_list):
    if action == "STOP(0)":
        return [[True, env.memory.copy(), cost, action_list]]
    elif depth < 0:
        return [[False, env.memory.copy(), cost, action_list]]
    else:
        node_name = action.split("(")[0]
        actions = model.get(node_name)

        if isinstance(actions, type(lambda x:0)):
            next_op = actions(None)
        else:
            next_op = actions.predict([env.get_observation().tolist()])[0]

        if next_op != "STOP(0)":
            action_name, args = next_op.split("(")[0], next_op.split("(")[1].replace(")", "")

            action_list.append((action_name, args))

            if args.isnumeric():
                args = int(args)

            cost += get_cost_from_env(env, action_name, str(args))

            env.act(action_name, args)

            return validation_recursive_tree(model, env, next_op, depth-1, cost, action_list)
        else:

            action_name, args = next_op.split("(")[0], next_op.split("(")[1].replace(")", "")

            action_list.append((action_name, args))

            if args.isnumeric():
                args = int(args)

            cost += get_cost_from_env(env, action_name, str(args))

            action_list.append(("STOP", "0"))
            return [[True, env.memory.copy(), cost, action_list]]


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
    parser.add_argument("--save", default=False, action="store_true", help="Save result to file")
    parser.add_argument("--to-stdout", default=False, action="store_true", help="Print results to stdout")

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
    costs = None
    total_actions = None
    length_actions = None
    method = None
    dataset = None
    results_filename = None

    if rank == 0:

        env = import_dyn_class(config.get("environment").get("name"))(
            **config.get("environment").get("configuration_parameters", {}),
            **config.get("validation").get("environment").get("configuration_parameters", {})
        )

        method="program"
        dataset=config.get("validation").get("dataset_name")

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
        date_time = '-validation-static-{}_{}_{}-{}_{}_{}.csv'.format(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])

        if args.save:
            results_filename = config.get("validation").get("save_results_name")+date_time
            #results_file = open(
            #    os.path.join(config.get("validation").get("save_results"), results_filename), "w"
            #)

        with open(args.model, "rb") as f:
            import dill as pickle
            model = pickle.load(f)

        # perform validation, not training
        env.validation = True

        reward = []
        costs = []
        total_actions = []
        length_actions = []

    iterations = min(int(config.get("validation").get("iterations")), len(env.data))

    for _ in tqdm(range(0, iterations//size), disable=args.to_stdout):

        if not args.single_core:
            env = comm.bcast(env, root=0)

        idx = env.prog_to_idx[args.task]

        _, state_index = env.start_task(idx)

        max_depth = env.max_depth_dict.get(1)

        next_action = "INTERVENE(0)"

        if args.tree:
            results = validation_recursive_tree(model, env, next_action, max_depth, 0, [])
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
                        reward.append(1)
                        costs.append(r[2])
                        total_actions.append(r[3])
                        length_actions.append(len(r[3]))
                        break
                    else:
                        reward.append(0)

        env.end_task()

    if rank == 0:

        # Create dataframe with the complete actions
        traces = []

        for k, trace in enumerate(total_actions):
            for p, a in trace:
                traces.append([
                    k, p, a
                ])

        t = pd.DataFrame(traces, columns=["id", "program", "argument"])

        #print("Correct:", reward)
        #print("Failures:", iterations-reward)
        #print("Mean/std cost: ", sum(costs)/len(costs), np.std(costs))
        #print("Mean/std length actions: ", sum(length_actions) / len(length_actions), np.std(length_actions))

        # Fix if they are empty
        costs = costs if costs else [0]
        length_actions = length_actions if length_actions else [0]

        if args.to_stdout:
            print(f"{method},{dataset},{np.mean(reward)},{1 - np.mean(reward)},{np.mean(costs)},{np.std(costs)},{np.mean(length_actions)},{np.std(length_actions)}")

        if args.save:
            #results_file.write(f"method,dataset,correct,wrong,mean_cost,std_cost,mean_length,std_length" + '\n')
            #results_file.write(f"{method},{dataset},{reward}, {1-np.mean(reward)}, {sum(costs)/len(costs)},{np.std(costs)},{sum(length_actions) / len(length_actions)},{np.std(length_actions)}" + '\n')
            #results_file.close()

            # Save sequences to file
            #df_sequences = []
            #for k, x in enumerate(traces):
            #    for p, a in x:
            #        df_sequences.append([k, env.get_program_from_index(p), a])

            # Create a dataframe and save sequences to disk
            if traces:
                best_sequences = pd.DataFrame(traces, columns=["id", "program", "arguments"])
                best_sequences.to_csv(
                    os.path.join(config.get("validation").get("save_results"),
                                     f"traces-{method}-{dataset}-{results_filename}"),
                        index=None)


