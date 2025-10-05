from typing import Dict, Any

from utils.tag_util import find_tags, extract_answer_tag
from verl.utils.reward_score.agent_rm.util import compute_tool_bonus

def parse_pointwise_score(score_str: str):
    score_str = extract_answer_tag(score_str)
    try:
        score = int(score_str)
    except:
        score = -1
    return score

def convert_score_to_reward(
    answer_score: int,
    ground_truth_score: int
):
    answer_score = answer_score-3
    ground_truth_score = ground_truth_score-3
    if answer_score == ground_truth_score:
        reward = 1
    elif answer_score * ground_truth_score > 0:
        reward = 0
    else:
        reward = -1
    return reward
    

def compute_pointwise_reward(
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
        # assume score in {1,2,3,4,5}
        answer_score = parse_pointwise_score(final_answer)
        parsed_ground_truth_score = parse_pointwise_score(ground_truth)
        reward = convert_score_to_reward(
            answer_score=answer_score,
            ground_truth_score=parsed_ground_truth_score
        )
        
    
    bonus = compute_tool_bonus(solution_str=solution_str,extra_info=extra_info)
    reward += bonus
    
    return reward
