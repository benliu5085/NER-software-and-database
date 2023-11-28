LR=$1
data=$2
tag=$3
INTER_HIDDEN=${4:-2048}

BASE=/home/b317l704
REPO_PATH=${BASE}/sentence_classifer_git
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_DIR=${REPO_PATH}/datasets/${data}
OUTPUT_BASE=${REPO_PATH}/outputs

BATCH=4
GRAD_ACC=4
DROPOUT=0.1
LR_MINI=3e-8
LR_SCHEDULER=polydecay
SPAN_WEIGHT=0.1
WARMUP=0
MAX_NORM=1.0
MAX_EPOCH=10
WEIGHT_DECAY=0.01
OPTIM=torch.adam
VAL_CHECK=0.2
PREC=16
WORKERS=6

OUTPUT_DIR=${OUTPUT_BASE}/${data}/lr${LR}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0 python3 mlp/trainer.py \
--data_dir ${DATA_DIR} \
--default_root_dir ${OUTPUT_DIR} \
--batch_size ${BATCH} \
--lr ${LR} \
--workers ${WORKERS} \
--weight_decay ${WEIGHT_DECAY} \
--warmup_steps ${WARMUP} \
--seed 0 \
--tag ${tag} \
--hidden_size ${INTER_HIDDEN} \
--dropout_rate ${DROPOUT} \
--optimizer ${OPTIM} \
--lr_scheduler ${LR_SCHEDULER} \
--lr_mini ${LR_MINI} \
--gpus="1" \
--precision=${PREC} \
--progress_bar_refresh_rate 1 \
--val_check_interval ${VAL_CHECK} \
--accumulate_grad_batches ${GRAD_ACC} \
--max_epochs ${MAX_EPOCH} \
--distributed_backend=ddp \
--gradient_clip_val ${MAX_NORM}
