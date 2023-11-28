DATA_DIR=$1
OUT_DIR=$2
CKPT=$3
TAG=$4

BASE=/home/b317l704
REPO_PATH=${BASE}/sentence_classifer_git
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

OUTPUT_BASE=${REPO_PATH}/${OUT_DIR}
MODEL_CKPT=${OUTPUT_BASE}/${CKPT}
HPARAMS_FILE=${OUTPUT_BASE}/lightning_logs/version_0/hparams.yaml

python3 ${REPO_PATH}/mlp/mlp_inference.py \
--data_dir ${DATA_DIR} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--output_dir ${OUTPUT_BASE} \
--tag ${TAG}
