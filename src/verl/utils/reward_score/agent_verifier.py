# Reward for tarining a agentic generatice reward model 

from typing import Dict, Any



from verl.utils.reward_score.agent_verifier.pairwise import compute_pairwise_reward
from verl.utils.reward_score.agent_verifier.pointwise import compute_pointwise_reward
from verl.utils.reward_score.agent_verifier.bool import compute_bool_reward, parse_bool_choice

from agentflow.utils.json_util import JsonUtil
from agentflow.utils.tag_util import find_tags

def _parse_correctness_reward(text: str, ground_truth: str):
    ground_truth = str(ground_truth)
    matches = find_tags(text, ["answer"])
    
    if matches:
        final_answer = matches[-1].body
    else:
        final_answer = None
        
    reward = 0
    if final_answer is None:
        reward = -1
    else:
        choice = parse_bool_choice(final_answer)
        parsed_ground_truth = parse_bool_choice(ground_truth)
        if parsed_ground_truth == "None":
            reward = 0
        if choice == parsed_ground_truth:
            reward = 1
        else:
            reward = -1
    return reward

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

    if data_source != "rm_bool":
        raise NotImplementedError(f"Reward fordatasource of {data_source} is not defined")
    
    
    stage = extra_info.get("stage",None)
    if not stage:
        score = compute_bool_reward(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
    elif stage == "plan":
        plan_score = extra_info.get("plan_score", 0)
        schemas = JsonUtil.parse_json("data_source")
        if schemas:
            format_score = 0.2
        else:
            format_score = 0
        score = format_score + plan_score
        
    elif stage == "subtask":
        subtask_gt = extra_info.get("subtask_gt","None")
        score = _parse_correctness_reward(solution_str,subtask_gt)
    elif stage == "review":
        score = _parse_correctness_reward(solution_str, ground_truth)
    else:
        raise ValueError(f"Stage {stage} is invalid")
    return score
        

def compute_agentic_reward_old(
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