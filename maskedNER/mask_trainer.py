import os
import re
import argparse
import logging
from collections import namedtuple
from typing import Dict
import time

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from tokenizers import BertWordPieceTokenizer
from torch import Tensor
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim import SGD

from maskedNER.mrc_ner_dataset import MRCNERDataset, collate_to_max_length
from maskedNER.bert_query_ner import BertQueryNER, BertQueryNerConfig
from utils.get_parser import get_parser
from utils.random_seed import set_random_seed

class BertLabeling(pl.LightningModule):
    def __init__( self, args ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        format = '%(asctime)s - %(name)s - %(message)s'
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
            self.args = args
            logging.basicConfig(format=format, filename=os.path.join(self.args.default_root_dir, "eval_result_log.txt"), level=logging.INFO)
        else:
            # eval mode
            TmpArgs = namedtuple("tmp_args", field_names=list(args.keys()))
            self.args = args = TmpArgs(**args)
            logging.basicConfig(format=format, filename=os.path.join(self.args.default_root_dir, "eval_test.txt"), level=logging.INFO)

        self.bert_dir = args.bert_config_dir
        self.data_dir = self.args.data_dir

        bert_config = BertQueryNerConfig.from_pretrained(args.bert_config_dir,
                                                         hidden_dropout_prob=args.bert_dropout,
                                                         attention_probs_dropout_prob=args.bert_dropout,
                                                         mrc_dropout=args.mrc_dropout,
                                                         classifier_act_func = args.classifier_act_func,
                                                         classifier_intermediate_hidden_size=args.classifier_intermediate_hidden_size)

        self.model = BertQueryNER.from_pretrained(args.bert_config_dir,
                                                  config=bert_config)
        logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)
        self.result_logger.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
        self.bce_loss = BCEWithLogitsLoss(reduction="none")
        self.optimizer = args.optimizer
        self.maskID = args.mask_ID

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--mrc_dropout", type=float, default=0.1,
                            help="mrc dropout rate")
        parser.add_argument("--bert_dropout", type=float, default=0.1,
                            help="bert dropout rate")
        parser.add_argument("--classifier_act_func", type=str, default="gelu")
        parser.add_argument("--classifier_intermediate_hidden_size", type=int, default=1024)
        parser.add_argument("--optimizer", choices=["adamw", "sgd", "torch.adam"], default="adamw",
                            help="loss type")
        parser.add_argument("--final_div_factor", type=float, default=1e4,
                            help="final div factor of linear decay scheduler")
        parser.add_argument("--lr_scheduler", type=str, default="onecycle", )
        parser.add_argument("--lr_mini", type=float, default=-1)
        parser.add_argument("--mask_ID", type=int, default=104)
        return parser

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon,)
        elif self.optimizer == "torch.adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)
        else:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9)
        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        if self.args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps/t_total),
                final_div_factor=self.args.final_div_factor,
                total_steps=t_total, anneal_strategy='linear'
            )
        elif self.args.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)
        elif self.args.lr_scheduler == "polydecay":
            if self.args.lr_mini == -1:
                lr_mini = self.args.lr / 5
            else:
                lr_mini = self.args.lr_mini
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, self.args.warmup_steps, t_total, lr_end=lr_mini)
        else:
            raise ValueError
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    def compute_loss(self, tokens, single_logits, single_labels):
        batch_size, seq_len = single_logits.size()
        masking = (tokens == self.maskID)
        single_loss = self.bce_loss(single_logits.view(-1), single_labels.view(-1).float())
        single_loss = (single_loss * masking.view(-1).float()).sum() / (masking.sum() + 1e-10)
        return single_loss

    def training_step(self, batch, batch_idx):
        tf_board_logs = { "lr": self.trainer.optimizers[0].param_groups[0]['lr']}

        tokens, token_type_ids, sample_idx, label_idx, single_labels = batch
        attention_mask = (tokens != 0).long()
        single_logits = self(tokens, attention_mask, token_type_ids)
        total_loss = self.compute_loss(tokens=tokens, single_logits=single_logits, single_labels=single_labels)
        tf_board_logs[f"train_loss"] = total_loss
        return {'loss': total_loss, 'log': tf_board_logs}

    def span_f1(self, tokens, single_preds, single_labels):
        gt = single_labels.bool()
        logits = single_preds > 0
        mask = (tokens == self.maskID)
        tp = ((logits.view(-1) & mask.view(-1)) & (gt.view(-1).bool() & mask.view(-1))).long().sum()
        fp = ((logits.view(-1) & mask.view(-1)) & (~gt.view(-1).bool() & mask.view(-1))).long().sum()
        fn = ((~logits.view(-1) & mask.view(-1)) & (gt.view(-1).bool() & mask.view(-1))).long().sum()
        tn = ((~logits.view(-1) & mask.view(-1)) & (~gt.view(-1).bool() & mask.view(-1))).long().sum()
        return torch.stack([tp, fp, fn])

    def validation_step(self, batch, batch_idx):
        output = {}
        tokens, token_type_ids, sample_idx, label_idx, single_labels = batch
        attention_mask = (tokens != 0).long()
        single_logits = self(tokens, attention_mask, token_type_ids)

        total_loss = self.compute_loss(tokens=tokens, single_logits = single_logits, single_labels=single_labels)
        output[f"val_loss"] = total_loss
        single_preds = single_logits > 0
        span_f1_stats = self.span_f1(tokens=tokens, single_preds=single_preds, single_labels=single_labels)
        output["span_f1_stats"] = span_f1_stats
        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        all_counts = torch.stack([x[f'span_f1_stats'] for x in outputs]).view(-1, 3).sum(0)
        span_tp, span_fp, span_fn = all_counts
        span_recall = span_tp / (span_tp + span_fn + 1e-10)
        span_precision = span_tp / (span_tp + span_fp + 1e-10)
        span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)
        tensorboard_logs[f"span_precision"] = span_precision
        tensorboard_logs[f"span_recall"] = span_recall
        tensorboard_logs[f"span_f1"] = span_f1
        self.result_logger.info(f"EVAL INFO -> current_epoch is: {self.trainer.current_epoch}, current_global_step is: {self.trainer.global_step} ")
        self.result_logger.info(f"EVAL INFO -> valid_f1 is: {span_f1}; precision: {span_precision}, recall: {span_recall}.")
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        output = {}
        tokens, token_type_ids, sample_idx, label_idx, single_labels = batch
        attention_mask = (tokens != 0).long()
        single_logits = self(tokens, attention_mask, token_type_ids)
        single_preds = single_logits > 0
        span_f1_stats = self.span_f1(tokens=tokens, single_preds=single_preds, single_labels=single_labels)
        output["span_f1_stats"] = span_f1_stats
        return output

    def test_epoch_end(self, outputs) -> Dict[str, Dict[str, Tensor]]:
        tensorboard_logs = {}
        all_counts = torch.stack([x[f'span_f1_stats'] for x in outputs]).view(-1, 3).sum(0)
        span_tp, span_fp, span_fn = all_counts
        span_recall = span_tp / (span_tp + span_fn + 1e-10)
        span_precision = span_tp / (span_tp + span_fp + 1e-10)
        span_f1 = span_precision * span_recall * 2 / (span_recall + span_precision + 1e-10)
        self.result_logger.info(f"TEST INFO -> test_f1 is: {span_f1} precision: {span_precision}, recall: {span_recall}")
        return {'log': tensorboard_logs}

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("dev")

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test")

    def get_dataloader(self, prefix="train") -> DataLoader:
        json_path = os.path.join(self.data_dir, f"mrc-ner.{prefix}")
        vocab_path = os.path.join(self.bert_dir, "vocab.txt")
        dataset = MRCNERDataset(json_path=json_path,
                                tokenizer=BertWordPieceTokenizer(vocab_path),
                                max_length=self.args.max_length,
                                pad_to_maxlen=False)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
            shuffle=True if prefix == "train" else False,
            collate_fn=collate_to_max_length
        )

        return dataloader

def find_best_checkpoint_on_dev(output_dir: str, log_file: str = "eval_result_log.txt", only_keep_the_best_ckpt: bool = False):
    with open(os.path.join(output_dir, log_file)) as f:
        log_lines = f.readlines()

    F1_PATTERN = re.compile(r"span_f1 reached \d+\.\d* \(best")
    CKPT_PATTERN = re.compile(r"saving model to \S+ as top")
    checkpoint_info_lines = []
    for log_line in log_lines:
        if "saving model to" in log_line:
            checkpoint_info_lines.append(log_line)
    best_f1_on_dev = 0
    best_checkpoint_on_dev = ""
    for checkpoint_info_line in checkpoint_info_lines:
        current_f1 = float(
            re.findall(F1_PATTERN, checkpoint_info_line)[0].replace("span_f1 reached ", "").replace(" (best", ""))
        current_ckpt = re.findall(CKPT_PATTERN, checkpoint_info_line)[0].replace("saving model to ", "").replace(
            " as top", "")

        if current_f1 >= best_f1_on_dev:
            if only_keep_the_best_ckpt and len(best_checkpoint_on_dev) != 0:
                os.remove(best_checkpoint_on_dev)
            best_f1_on_dev = current_f1
            best_checkpoint_on_dev = current_ckpt

    return best_f1_on_dev, best_checkpoint_on_dev


def main():
    """main"""
    parser = get_parser()
    parser = BertLabeling.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    set_random_seed(args.seed)

    print(args.seed, torch.initial_seed())
    print(args.mask_ID)

    model = BertLabeling(args)
    if args.pretrained_checkpoint:
        model.load_state_dict(torch.load(args.pretrained_checkpoint,
                                         map_location=torch.device('cpu'))["state_dict"])

    checkpoint_callback = ModelCheckpoint(
        filepath=args.default_root_dir,
        save_top_k=args.max_keep_ckpt,
        verbose=True,
        monitor="span_f1",
        period=-1,
        mode="max",
    )
    trainer = Trainer.from_argparse_args(
        args,
        checkpoint_callback=checkpoint_callback,
        deterministic=True,
        default_root_dir=args.default_root_dir
    )

    trainer.fit(model)

    # after training, use the model checkpoint which achieves the best f1 score on dev set to compute the f1 on test set.
    best_f1_on_dev, path_to_best_checkpoint = find_best_checkpoint_on_dev(args.default_root_dir, )
    model.result_logger.info("=&" * 20)
    model.result_logger.info(f"Best F1 on DEV is {best_f1_on_dev}")
    model.result_logger.info(f"Best checkpoint on DEV set is {path_to_best_checkpoint}")
    checkpoint = torch.load(path_to_best_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.result_logger.info("=&" * 20)


if __name__ == '__main__':
    main()
