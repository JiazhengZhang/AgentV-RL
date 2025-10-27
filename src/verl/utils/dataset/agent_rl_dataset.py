from __future__ import annotations
from typing import Dict, Any, List, Tuple, Union
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.common.agent_rl_utils import MultiStagePlan

def multistage_collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert samples, "empty batch"
    batch_meta = samples[0]["__batch_meta__"]
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    data: Dict[str, Union[torch.Tensor, np.ndarray]] = {}
    
    for sample in samples:
        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)
                
    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)
            
    data.update(tensors)
    data.update(non_tensors)
    
    result_dict = {"meta_info": batch_meta, **data}

    return result_dict

class StageExpandedRLHFDataset(Dataset):
    """Provide a virtual multistage view of the original RLHF dataset

    """
    
    def __init__(self, base_dataset: RLHFDataset, plan: MultiStagePlan):
        self.base = base_dataset
        self.plan = plan
        self._len = len(plan.base_idx_arr)  

    def __len__(self): 
        return self._len

    def __getitem__(self, virtual_idx: int) -> Dict[str, Any]:
        meta = self.plan.decode(virtual_idx)       
        base_idx = meta["base_idx"]    
        row  = self.base[base_idx]            

        out: Dict[str, Any] = dict(row)

        out["index"] = row.get("index", meta["base_idx"])
        out["base_idx"] = base_idx
        
        out["__batch_meta__"] = {
            "stage": meta["stage"],
            "group_id": meta["group_id"],
            "repeat_id": meta["repeat_id"],
            "is_group_final": meta["is_group_final"],
            "subtask_slot": meta["subtask_slot"],
        }
        return out
    
