LR=$1
WHICH_BERT=$2
FILE=$3

BASE=/home/b317l704
REPO_PATH=${BASE}/sentence_classifer_git
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_DIR=${REPO_PATH}/datasets/DS0
BERT_DIR=${REPO_PATH}/${WHICH_BERT}
OUTPUT_BASE=${REPO_PATH}/outputs

BATCH=4
GRAD_ACC=4
BERT_DROPOUT=0.1
MRC_DROPOUT=0.3
LR_MINI=3e-8
LR_SCHEDULER=polydecay
SPAN_WEIGHT=0.1
WARMUP=0
MAX_LEN=200
MAX_NORM=1.0
MAX_EPOCH=10
INTER_HIDDEN=2048
WEIGHT_DECAY=0.01
OPTIM=torch.adam
VAL_CHECK=0.2
PREC=16
SPAN_CAND=pred_and_gold
WORKERS=6

OUTPUT_DIR=${OUTPUT_BASE}/DS0/${FILE}/lr${LR}_maxlen${MAX_LEN}
mkdir -p ${OUTPUT_DIR}


python3 ${REPO_PATH}/unmaskedNER/unmask_trainer.py \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--batch_size ${BATCH} \
--gpus="1" \
--precision=${PREC} \
--progress_bar_refresh_rate 1 \
--lr ${LR} \
--val_check_interval ${VAL_CHECK} \
--accumulate_grad_batches ${GRAD_ACC} \
--default_root_dir ${OUTPUT_DIR} \
--mrc_dropout ${MRC_DROPOUT} \
--bert_dropout ${BERT_DROPOUT} \
--max_epochs ${MAX_EPOCH} \
--span_loss_candidates ${SPAN_CAND} \
--weight_span ${SPAN_WEIGHT} \
--warmup_steps ${WARMUP} \
--distributed_backend=ddp \
--max_length ${MAX_LEN} \
--gradient_clip_val ${MAX_NORM} \
--weight_decay ${WEIGHT_DECAY} \
--optimizer ${OPTIM} \
--lr_scheduler ${LR_SCHEDULER} \
--classifier_intermediate_hidden_size ${INTER_HIDDEN} \
--flat \
--lr_mini ${LR_MINI} \
--workers ${WORKERS} \
--seed 0
