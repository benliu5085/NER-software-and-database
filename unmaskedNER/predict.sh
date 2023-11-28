OUT_DIR=$1 ## outputs/DS0/
CKPT=$2
WHICH_BERT=$3

BASE=/home/b317l704
REPO_PATH=${BASE}/sentence_classifer_git
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_DIR=${REPO_PATH}/datasets/DS0
BERT_DIR=${REPO_PATH}/${WHICH_BERT}
MAX_LEN=200

# find best checkpoint on dev in ${OUTPUT_DIR}/train_log.txt
OUTPUT_BASE=${REPO_PATH}/${OUT_DIR}
MODEL_CKPT=${OUTPUT_BASE}/${CKPT}
HPARAMS_FILE=${OUTPUT_BASE}/lightning_logs/version_0/hparams.yaml


python3 ${REPO_PATH}/unmaskedNER/unmask_inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--output_dir ${OUTPUT_BASE} \
--test_data DS0
