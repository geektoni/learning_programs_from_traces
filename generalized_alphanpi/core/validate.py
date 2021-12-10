from generalized_alphanpi.utils import import_dyn_class, get_cost_from_env, get_cost_from_tree

import numpy as np

from argparse import ArgumentParser
import torch
import yaml

import time
import os
from tqdm import tqdm

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="Path to the model we want to visualize.")
    parser.add_argument("task", type=str, help="Task we want to execute")
    parser.add_argument("--config", type=str, help="Path to the file with the experiment configuration")
    parser.add_argument("--save", default=False, action="store_true", help="Save result to file")

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
    mcts.number_of_simulations = 5
    mcts.env.validation = True

    mcts_rewards_normalized = []
    mcts_rewards = []
    mcts_cost = []
    mcts_length = []
    failures = 0.0

    ts = time.localtime(time.time())
    date_time = '-{}-{}_{}_{}-{}_{}_{}.csv'.format(args.task, ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])

    results_file = None
    if args.save:
        results_filename = config.get("validation").get("save_results_name")+date_time
        results_file = open(
            os.path.join(config.get("validation").get("save_results"), results_filename), "w"
        )

    iterations = min(int(config.get("validation").get("iterations")), len(env.data))
    for _ in tqdm(range(0, iterations)):

        trace, root_node = mcts.sample_execution_trace()

        if trace.rewards[0] > 0:
            cost, length = get_cost_from_tree(env, root_node)
            mcts_rewards.append(trace.rewards[0].item())
            mcts_rewards_normalized.append(1.0)
            mcts_cost.append(cost)
            mcts_length.append(length)
        else:
            mcts_rewards.append(0.0)
            mcts_rewards_normalized.append(0.0)
            failures += 1

    mcts_rewards_normalized_mean = np.mean(np.array(mcts_rewards_normalized))
    mcts_rewards_normalized_std = np.std(np.array(mcts_rewards_normalized))
    mcts_rewards_mean = np.mean(np.array(mcts_rewards))
    mcts_rewards_std = np.std(np.array(mcts_rewards))
    mcts_cost_mean = np.mean(mcts_cost)
    mcts_cost_std = np.std(mcts_cost)
    mcts_length_mean = np.mean(mcts_length)
    mcts_length_std = np.std(mcts_length)

    complete = f"{mcts_rewards_mean},{mcts_rewards_normalized_mean},{mcts_rewards_std},{mcts_rewards_normalized_std},{mcts_cost_mean},{mcts_cost_std},{mcts_length_mean},{mcts_length_std}"

    print("mcts_reward_mean,mcts_reward_normalized_mean,mcts_rewards_std,mcts_rewards_normalized_std,mcts_cost_mean,mcts_cost_std,mcts_length_mean,mcts_length_std")
    print("Complete:", complete)
    print("Failures:", failures)

    # Save results to a file
    if args.save:
        results_file.write("mcts_reward_mean,mcts_reward_normalized_mean,mcts_rewards_std,mcts_rewards_normalized_std,mcts_cost_mean,mcts_cost_std,mcts_length_mean,mcts_length_std\n")
        results_file.write(complete + '\n')
        results_file.close()
