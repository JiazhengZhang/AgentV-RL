from typing import Dict, Any

from agentflow.utils.json_util import JsonUtil
from agentflow.utils.tag_util import find_tags
from agentflow.utils.math.answer_parser import grade_answer_verl



def compute_math_reward(
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

    
    assert ground_truth is not None, "GT cannot be None"
    extra_info = extra_info or {}

    entry = grade_answer_verl(solution_str, ground_truth)
    
    if entry["correct"] is True:
        score = 1
    else:
        score = -1
    return score

