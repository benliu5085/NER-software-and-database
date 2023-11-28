import os
import torch
import argparse
from torch.utils.data import DataLoader
from utils.random_seed import set_random_seed
from tokenizers import BertWordPieceTokenizer
set_random_seed(0)
from maskedNER.mrc_ner_dataset import MRCNERDataset, collate_to_max_length
from maskedNER.mask_trainer import BertLabeling

BS = 32
workers=6
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
    parser.add_argument("--data_dir", type=str, default="/panfs/panfs.ittc.ku.edu/scratch/ben0522/test/mrc-for-flat-nested-ner/datasets/gold")
    parser.add_argument("--bert_dir", type=str, default="/panfs/panfs.ittc.ku.edu/scratch/ben0522/test/dice_loss_for_NLP/bert/wwm_cased_L-24_H-1024_A-16")
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--model_ckpt", type=str, default="")
    parser.add_argument("--hparams_file", type=str, default="")
    parser.add_argument("--maskID", type=int, choices=[103, 104], default=104)
    return parser


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
    for pmcid in pick_list:
        data_loader, data_tokenizer = get_dataloader(args, pmcid)
        for batch in data_loader:
            tokens, token_type_ids, sample_idx, label_idx, single_labels = batch
            attention_mask = (tokens != 0).long()
            single_logits = trained_mrc_ner_model.model(tokens, attention_mask=attention_mask, token_type_ids=token_type_ids)
            mask = (tokens == args.maskID)
            for sid in range(single_logits.shape[0]):
                score = single_logits[sid,:][mask[sid,:]]
                print(f"{pmcid}\t{int(sample_idx[sid])}.{int(label_idx[sid])}\t{score}")

if __name__ == "__main__":
    main()
