SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# train eagle3 3-layer for llama3.1-8b
NUM_GPUS=${1:-1}
TP_SIZE=${1:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path nreHieW/Llama-3.1-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3_3layer.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfectblend_train.jsonl \
    --eval-data-path $ROOT_DIR/cache/dataset/perfectblend_test.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/llama3-8b-eagle3-perfectblend-3layer-flow-2e-4 \
    --num-epochs 2 \
    --warmup-ratio 0.05 \
    --batch-size 1 \
    --eval-batch-size 1 \
    --draft-accumulation-steps 64 \
    --tp-size $TP_SIZE \
    --learning-rate 2e-4 \
    --max-length 2048 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache \
    --attention-backend sdpa \
    --target-model-backend custom \
    --log-interval 10 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-ratio 0.2 \
    --ttt-length 8 \
    --sglang-mem-fraction-static 0.5 \
    --resume \
    --report-to wandb \
    --wandb-resume-from lq1gr9u2?_step=12000