from typing import Dict, Any
import warnings

from agentflow.utils.tag_util import find_tags

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
    true_answer = choice_str
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
    subtasks_matches = find_tags(solution_str, ["subtasks"])
    if not subtasks_matches:
        return -1
    subtask_end = subtasks_matches[-1].end
    
    
    if matches:
        final_answer_pos = matches[-1].start
        if final_answer_pos >= subtask_end:
            final_answer = matches[-1].body
        else:
            final_answer = None
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
            
    return reward
    


