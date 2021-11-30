from generalized_alphanpi.utils import import_dyn_class

import numpy as np

from argparse import ArgumentParser
import torch
import yaml

import time
import os
from tqdm import tqdm

import dill

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="Path to the automa model we want to validate.")
    parser.add_argument("task", type=str, help="Task we want to execute")
    parser.add_argument("--config", type=str, help="Path to the file with the experiment configuration")

    args = parser.parse_args()
    config = yaml.load(open(args.config),Loader=yaml.FullLoader)

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
    date_time = '-{}-{}_{}_{}-{}_{}_{}.csv'.format(args.task, ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])

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
    for _ in tqdm(range(0, iterations)):

        _, state_index = env.start_task(idx)

        next_action = "INTERVENE(0)"
        while next_action != "STOP(0)":

            observation = env.memory.copy()

            node_name = next_action.split("(")[0]

            actions = model.get(node_name)

            found=False
            for a, conditions in actions.items():
                results = []

                parsed_observation = {c: v for c, v in zip(env.parsed_columns, env.parse_observation(observation))}

                for c in conditions:
                    tmp = [f(parsed_observation) for f in c]
                    results.append(all(tmp))
                results = any(results)

                if results:
                    next_action = a
                    found=True
                    break

            if not found:
                print("No condition satisfied!")
                failures += 1
                break

            action_name, args = next_action.split("(")[0], next_action.split("(")[1].replace(")", "")

            if args.isnumeric():
                args = int(args)

            env.act(action_name, args)
            observation = env.memory.copy()

        if env.prog_to_postcondition[env.get_program_from_index(idx)](None, None):
            reward += 1
        else:
            failures += 1

        env.end_task()


    print("Correct:", reward)
    print("Failures:", failures)

    #results_file.write(complete + '\n')
    #results_file.close()
