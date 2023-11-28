import os
import torch
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from utils.random_seed import set_random_seed
from utils.bmes_decode import bmes_decode
from tokenizers import BertWordPieceTokenizer
set_random_seed(0)
from unmaskedNER.mrc_ner_dataset import MRCNERDataset, collate_to_max_length
from unmaskedNER.unmask_trainer import BertLabeling

BS = 32
workers=6

def clean(labels):
    N = len(labels)
    ans = ["O"] * N
    idx = 0
    while idx < N:
        curr = labels[idx]
        if curr == 'O' or curr[0] == 'S':
            ans[idx] = labels[idx]
            idx += 1
            continue
        if curr[0] == 'B': # clean until E, might jump a lot of postion
            jdx = idx+1
            if jdx == N: # last B
                ans[idx] = 'S'+labels[idx][1:]
                idx += 1
                continue
            while jdx < N and labels[jdx][0] in "BMS":
                jdx += 1
            if jdx == N:
                ans[idx] = curr
                for tmp_i in range(idx+1, jdx):
                    ans[tmp_i] = 'M'+labels[tmp_i][1:]
                ans[jdx-1] = 'E'+labels[jdx-1][1:]
                idx = jdx+1
                continue
            if labels[jdx][0] == 'E': # end with E, legit format
                ans[idx] = curr
                for tmp_i in range(idx+1, jdx):
                    ans[tmp_i] = 'M'+labels[jdx][1:]
                ans[jdx] = labels[jdx]
                idx = jdx+1
                continue
            else: # clean all non S output
                for tmp_i in range(idx, jdx+1):
                    if labels[tmp_i][0] == 'S':
                        ans[tmp_i] = labels[tmp_i]
                idx = jdx+1
                continue
        idx += 1
    return ans

def extract_flat_spans( start_pred, end_pred, match_pred, start_label_mask, end_label_mask, cutoff = 0):
    pseudo_input = "a"
    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()
    start_pred = (start_pred > cutoff)[:]
    end_pred = (end_pred > cutoff)[:]
    match_pred = (match_pred > cutoff)[:,:]
    bmes_labels = ["O"] * len(start_pred)
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and start_label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and end_label_mask[idx]]
    for start_item in start_positions:
        bmes_labels[start_item] = f"B-{pseudo_input}"
    for end_item in end_positions:
        bmes_labels[end_item] = f"E-{pseudo_input}"
    for tmp_start in start_positions:
        tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        if match_pred[tmp_start][tmp_end]:
            if tmp_start != tmp_end:
                for i in range(tmp_start+1, tmp_end):
                    bmes_labels[i] = f"M-{pseudo_input}"
            else:
                bmes_labels[tmp_end] = f"S-{pseudo_input}"
    bmes_labels = clean(bmes_labels)
    tags = bmes_decode([(pseudo_input, label) for label in bmes_labels])
    return [(entity.begin, entity.end, entity.tag) for entity in tags]


def get_dataloader(config, data_prefix="test"):
    data_path = os.path.join(config.data_dir, f"mrc-ner.{data_prefix}")
    vocab_path = os.path.join(config.bert_dir, "vocab.txt")
    data_tokenizer = BertWordPieceTokenizer(vocab_path)
    dataset = MRCNERDataset(json_path=data_path,
                            tokenizer=data_tokenizer,
                            max_length=config.max_length,
                            pad_to_maxlen=False)
    dataloader = DataLoader(dataset=dataset, batch_size=BS, shuffle=False, num_workers=workers, collate_fn=collate_to_max_length)
    return dataloader, data_tokenizer

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="inference the model output.")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--bert_dir", type=str, default="")
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--model_ckpt", type=str, default="")
    parser.add_argument("--hparams_file", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--test_data", type=str, default="DS0")
    return parser

# argv = [
# "--data_dir",     "/home/b317l704/sentence_classifer_git/datasets/DS0",
# "--bert_dir",     "/home/b317l704/sentence_classifer_git/biobert_v1.1_pubmed",
# "--model_ckpt",   "/home/b317l704/sentence_classifer_git/outputs/DS0/bioBERT/lr1e-5_maxlen200/epoch=7_v0.ckpt",
# "--hparams_file", "/home/b317l704/sentence_classifer_git/outputs/DS0/bioBERT/lr1e-5_maxlen200/lightning_logs/version_0/hparams.yaml",
# "--output_dir",   "/home/b317l704/sentence_classifer_git/outputs/DS0/bioBERT/lr1e-5_maxlen200"
# ]
def main():
    parser = get_parser()
    args = parser.parse_args()
    trained_mrc_ner_model = BertLabeling.load_from_checkpoint(
        checkpoint_path=args.model_ckpt,
        hparams_file=args.hparams_file,
        map_location=None,
        batch_size=BS,
        max_length=args.max_length,
        workers=workers)
    pick_list = []
    list_path = os.path.join(args.data_dir, "example.txt")
    with open(list_path, 'r') as fin:
        for l in fin:
            pick_list.append(l.strip())
    # load token
    vocab_path = os.path.join(args.bert_dir, "vocab.txt")
    with open(vocab_path, "r") as f:
        subtokens = [token.strip() for token in f.readlines()]
    idx2tokens = {}
    for token_idx, token in enumerate(subtokens):
        idx2tokens[token_idx] = token
    data = []
    for pmcid in pick_list:
        data_loader, data_tokenizer = get_dataloader(args, pmcid)
        for batch in data_loader:
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx = batch
            attention_mask = (tokens != 0).long()
            start_logits, end_logits, span_logits = trained_mrc_ner_model.model(tokens, attention_mask=attention_mask, token_type_ids=token_type_ids)
            for sid in range(start_logits.shape[0]):
                subtokens_idx_lst = tokens[sid].tolist()
                subtokens_lst = [idx2tokens[item] for item in subtokens_idx_lst]
                entities_info = extract_flat_spans( start_logits[sid], end_logits[sid], span_logits[sid], start_label_mask[sid], end_label_mask[sid])
                entity_lst = []
                if len(entities_info) != 0:
                    for entity_info in entities_info:
                        start, end = entity_info[0], entity_info[1]
                        entity_string = " ".join(subtokens_lst[start: end])
                        entity_string = entity_string.replace(" ##", "")
                        entity_lst.append((start, end, entity_string))
                data.append({"dataset": pmcid,
                             "sample_id": sample_idx[sid].item(),
                             "label_id": label_idx[sid].item(),
                             "entity_lst": entity_lst})
    pd.DataFrame(data).to_csv(f"{args.output_dir}/{args.test_data}.csv", index=None)
    
if __name__ == "__main__":
    main()
