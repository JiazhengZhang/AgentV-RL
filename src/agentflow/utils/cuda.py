import os
import logging
import time

def parse_visible_devices() -> list[int]:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not vis:
        return []
    return [int(x) for x in vis.split(",") if x.strip()]

def get_gpu_memory_info(gpu_id: int) -> dict:
    out = os.popen(
        f"nvidia-smi --id={gpu_id} "
        f"--query-gpu=memory.free,memory.total "
        f"--format=csv,noheader,nounits"
    ).read().strip()
    free_mb, total_mb = map(int, out.split(","))
    return {"free_mb": free_mb, "total_mb": total_mb}

def all_devices_below_threshold(devices: list[int], threshold: float) -> bool:
    for gpu in devices:
        info = get_gpu_memory_info(gpu)
        used_frac = 1.0 - info["free_mb"] / info["total_mb"]
        print(f"GPU {gpu}: used={used_frac:.3f}")
        if used_frac >= threshold:
            return False
    return True

def wait_device(threshold: float = 0.05):
    visible = parse_visible_devices()
    print(f"Visible devices: {visible}")
    while True:
        if all_devices_below_threshold(visible, threshold):
            print(f"GPUs are free, cancel waiting.")
            break
        else:
            print(f"GPUS are not free...Waiting 5s for next try...")
            time.sleep(5)