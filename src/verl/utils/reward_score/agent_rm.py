# Reward for tarining a agentic generatice reward model 

from typing import Dict, Any



from verl.utils.reward_score.agent_rm.pairwise import compute_pairwise_reward
from verl.utils.reward_score.agent_rm.pointwise import compute_pointwise_reward
from verl.utils.reward_score.agent_rm.bool import compute_bool_reward

def compute_agentic_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str,Any] = None,
) -> float:
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.
    
    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "rm_pairwise":
        return compute_pairwise_reward(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    elif data_source == "rm_score":
        return compute_pointwise_reward(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    elif data_source == "rm_bool":
        return compute_bool_reward(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    else:
        raise NotImplementedError(f"Reward fordatasource of {data_source} is not defined")