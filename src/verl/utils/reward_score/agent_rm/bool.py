from typing import Dict, Any
import warnings

from utils.tag_util import find_tags, extract_answer_tag
from verl.utils.reward_score.agent_rm.util import compute_tool_bonus

ALLOWED_MODEL_TAGS = {
    "1":[
      "true",
      "True",
        
    ],
    "2":[
       "false",
       "False"
    ]
}

def parse_bool_choice(choice_str: str) -> str:
    true_answer = extract_answer_tag(choice_str)
    if true_answer.strip() in ALLOWED_MODEL_TAGS["1"]:
        return "1"
    elif true_answer.strip() in ALLOWED_MODEL_TAGS["2"]:
        return "2"
    else:
        return "None"

def compute_bool_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str,Any] = None,
) -> float:
    matches = find_tags(solution_str, ["answer"])
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
            warnings.warn(f"Ground truth {ground_truth} with source {data_source} cannot be parsed for bool reward computation")
            reward = 0
        if choice == parsed_ground_truth:
            reward = 1
        else:
            reward = -1
            
    bonus = compute_tool_bonus(solution_str=solution_str,extra_info=extra_info)
    reward += bonus
    
    return reward
    


