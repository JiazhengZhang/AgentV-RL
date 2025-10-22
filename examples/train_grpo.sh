set -x

export WANDB_MODE=offline
export WANDB_DIR=/root/workspace/agent-rm/wandb

export CUDA_VISIBLE_DEVICES=5,6
export RAY_TEMP_DIR=/mnt/data/ray_temp
export NCCL_TIMEOUT=3600  
export HYDRA_FULL_ERROR=1

# verl path in src
SRC_DIR="/root/workspace/agent-rm/Agent-Verifier/src"



export PYTHONPATH="${SRC_DIR}"
echo "PYTHONPATH = $PYTHONPATH"



PROJECT_NAME=rm_grpo_1020 # project name
EXP_NAME=grpo-2000-qwen3-4b-test-1 # exp name

ACTOR_MODEL_PATH=/root/workspace/agent-rm/models/qwen3-4b

SAVE_BASE_DIR=/root/workspace/agent-rm/checkpoints
PROJECT_DIR=${PROJECT_NAME}/${EXP_NAME}

train_files=/root/workspace/agent-rm/datasets/polaris/rl-1011/rl_1011-3000.train.parquet
test_files=/root/workspace/agent-rm/datasets/polaris/rl-1011/rl_1011-3000.val.parquet

ROLLOUT_N=4

CONFIG_DIR=/root/workspace/agent-rm/Agent-Verifier/src/verl/config/RL
CONFIG_NAME=grpo_verifier # verl config name
REWAED_FN_PATH=${SRC_DIR}/verl/utils/reward_score/agent_verifier.py
AGENT_CONFIG_PATH=/root/workspace/agent-rm/Agent-Verifier/config/train_grpo.yaml

n_gpus_per_node=2
nnodes=1


ray stop
ray start --head \
    --num-gpus=2 \
    --plasma-directory=/tmp/ray_plasma \
    --object-store-memory=274877906944



# You need modify log path
python3 -m verl.trainer.main_ppo --config-path=$CONFIG_DIR --config-name=$CONFIG_NAME\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=12 \
    data.max_prompt_length=2400 \
    data.max_response_length=5600 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=12800 \
    actor_rollout_ref.actor.checkpoint.save_contents='["hf_model"]' \
    actor_rollout_ref.extra.agent_config_path=$AGENT_CONFIG_PATH\
    custom_reward_function.path=$REWAED_FN_PATH \
    custom_reward_function.name=compute_agentic_reward \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_local_dir=${SAVE_BASE_DIR}/${PROJECT_DIR} \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.val_before_train=False \
    trainer.nnodes=$nnodes \
    trainer.save_freq=60 \
    trainer.test_freq=60 \
    trainer.rollout_data_dir=${SAVE_BASE_DIR}/${PROJECT_DIR}/rollout_data  \
    trainer.validation_data_dir=${SAVE_BASE_DIR}/${PROJECT_DIR}/validation_data \
    trainer.verl_dir=$SRC_DIR \
    trainer.total_epochs=1 $@ \
     2>&1 | tee -a /root/workspace/agent-rm/log/${PROJECT_NAME}_${EXP_NAME}.log
