import torch
from generalized_alphanpi.core import ExecutionTrace
from generalized_alphanpi.environments.environment import Environment

import importlib

def import_dyn_class(path: str):
    """
    Import dynamically a python class which can be then used
    :param path: path to the class
    :return: the class which can be instantiated
    """

    class_name = path.split(".")[-1]
    module_name = ".".join(path.split(".")[:-1])

    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)

    return class_

def print_trace(trace: ExecutionTrace, env: Environment) -> None:
    print(f"Trace ({len(trace.previous_actions)}):")
    for p in trace.previous_actions[1:]:
        print(f"\t {env.get_program_from_index(p)}")

def fix_policy(policy):
    """
    This fix potential issues which happens withing the policy. Namely, NaN probabilities
    or negative probabilites (which should not happen anyway).
    :param policy: the policy we need to fix
    :return: a safe version of the policy
    """

    safe_policy = torch.where(torch.isnan(policy), torch.zeros_like(policy), policy)
    safe_policy = torch.max(safe_policy, torch.tensor([0.]))

    # This means that we changed something. Therefore we need to recompute
    # the policy taking into account the corrected values
    if safe_policy.sum() != 1.0 and safe_policy.sum() != 0.0:
        safe_policy = safe_policy / safe_policy.sum()

    return safe_policy

def compute_q_value(node, qvalue_temperature):
    if node.visit_count > 0.0:
        values = torch.FloatTensor(node.total_action_value)
        softmax = torch.exp(qvalue_temperature * values)
        softmax = softmax / softmax.sum()
        q_val_action = float(torch.dot(softmax, values))
    else:
        q_val_action = 0.0
    return q_val_action