#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# description:
# NOTICE:
# Please make sure tensorflow
#


BERT_BASE_DIR=/home/b317l704/biobert/biobert_v1.1_pubmed

transformers-cli convert --model_type bert \
  --tf_checkpoint ${BERT_BASE_DIR}/bert_model.ckpt \
  --config ${BERT_BASE_DIR}/bert_config.json \
  --pytorch_dump_output ${BERT_BASE_DIR}/pytorch_model.bin

cp ${BERT_BASE_DIR}/bert_config.json ${BERT_BASE_DIR}/config.json
