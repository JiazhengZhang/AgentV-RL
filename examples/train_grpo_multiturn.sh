set -x

# export WANDB_MODE=offline
export WANDB_DIR=/root/workspace/agent-rm/wandb

export CUDA_VISIBLE_DEVICES=1,2,3,4
export NCCL_TIMEOUT=3600  
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


# verl path in src
SRC_DIR="/root/workspace/agent-rm/Agent-Verifier/src"



export PYTHONPATH="${SRC_DIR}"
echo "PYTHONPATH = $PYTHONPATH"



PROJECT_NAME=rm_grpo_multiturn_1111 # project name
EXP_NAME=grpo-multistage-qwen2.5-7b-1 # exp name

ACTOR_MODEL_PATH=/root/workspace/agent-rm/models/Qwen-2.5-7B-Instruct

SAVE_BASE_DIR=/root/workspace/agent-rm/checkpoints
PROJECT_DIR=${PROJECT_NAME}/${EXP_NAME}

train_files=/root/workspace/agent-rm/datasets/polaris/rl-1111/rl_multiturn_1111-1_train.parquet
test_files=/root/workspace/agent-rm/datasets/polaris/rl-1111/rl_multiturn_1111-1_val.parquet

ROLLOUT_N=5

CONFIG_DIR=/root/workspace/agent-rm/Agent-Verifier/src/verl/config/RL
CONFIG_NAME=grpo_verifier # verl config name
REWAED_FN_PATH=${SRC_DIR}/verl/utils/reward_score/agent_verifier.py
AGENT_CONFIG_PATH=/root/workspace/agent-rm/Agent-Verifier/config/train_grpo_local.yaml
ENABLE_THINKING=False

n_gpus_per_node=4
nnodes=1


ray stop
ray start --head \
    --num-gpus=4 \
    --plasma-directory=/tmp/ray_plasma \
    --object-store-memory=274877906944



# You need modify log path
python3 -m verl.trainer.main_ppo --config-path=$CONFIG_DIR --config-name=$CONFIG_NAME\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=8 \
    data.max_prompt_length=3200 \
    data.max_response_length=12000 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.per_round_max_tokens=4096 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=30000 \
    actor_rollout_ref.actor.checkpoint.save_contents='["hf_model"]' \
    actor_rollout_ref.extra.agent_config_path=$AGENT_CONFIG_PATH\
    actor_rollout_ref.extra.use_multiturn_wrapper=True \
    actor_rollout_ref.extra.enable_thinking=${ENABLE_THINKING} \
    custom_reward_function.path=$REWAED_FN_PATH \
    custom_reward_function.name=compute_agentic_reward \
    algorithm.use_kl_in_reward=False \
    trainer.user_multi_stage=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_local_dir=${SAVE_BASE_DIR}/${PROJECT_DIR} \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.val_before_train=False \
    trainer.nnodes=$nnodes \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.rollout_data_dir=${SAVE_BASE_DIR}/${PROJECT_DIR}/rollout_data  \
    trainer.validation_data_dir=${SAVE_BASE_DIR}/${PROJECT_DIR}/validation_data \
    trainer.verl_dir=$SRC_DIR \
    trainer.total_epochs=1 $@ \
     2>&1 | tee -a /root/workspace/agent-rm/log/${PROJECT_NAME}_${EXP_NAME}.log
