set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline

export PYTHONPATH=/root/workspace/agent-rm/Agent-Verifier/src:$PYTHONPATH
echo $PYTHONPATH

nproc_per_node=4
CONFIG_PATH=/root/workspace/agent-rm/Agent-Verifier/src/verl/config/SFT

PROJECT_NAME=rm-sft-1110
EXP_NAME=sft-qwen3-4b-agentic-think-1

TRAIN_FILES=/root/workspace/agent-rm/datasets/polaris/sft1108/sft_multiturn_1109.parquet
VAL_FILES=/root/workspace/agent-rm/datasets/polaris/sft1108/sft_multiturn_1109.parquet

SAVE_DIR=/root/workspace/agent-rm/checkpoints

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=123456 \
     -m verl.trainer.fsdp_sft_trainer \
    --config-path $CONFIG_PATH \
    --config-name multiturn_sft_trainer.yaml \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=25600 \
    model.partial_pretrain=/root/workspace/agent-rm/models/qwen3-4b \
    optim.lr=5e-6 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=1 \
    trainer.save_freq=65 \
    trainer.test_freq=500 \
    trainer.default_local_dir=${SAVE_DIR}/${PROJECT_NAME}/${EXP_NAME} \
    trainer.total_epochs=2 $@ \
    2>&1 | tee -a /root/workspace/agent-rm/log/${PROJECT_NAME}_${EXP_NAME}.log
    