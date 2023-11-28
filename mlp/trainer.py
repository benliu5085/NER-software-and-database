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

from mlp.dataset import fasttextDataset, tfidfDataset, bioNerDSDataset, collate
from mlp.model import MLPModel
from utils.random_seed import set_random_seed

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--data_dir", type=str, required=True, help="data dir")
    parser.add_argument("--max_keep_ckpt", default=3, type=int, help="the number of keeping ckpt max.")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps used for scheduler.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--seed", default=0, type=int, help="set random seed for reproducing results.")
    parser.add_argument("--tag", choices=["tfidf", "fasttext", "bionerds"], default="tfidf")
    parser.add_argument("--act_func", type=str, default="gelu")
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=["adamw", "sgd", "torch.adam"], default="torch.adam",
                                help="loss type")
    parser.add_argument("--final_div_factor", type=float, default=1e4,
                                help="final div factor of linear decay scheduler")
    parser.add_argument("--lr_scheduler", type=str, default="onecycle", )
    parser.add_argument("--lr_mini", type=float, default=-1)
    return parser


class SimpleModel(pl.LightningModule):
    def __init__( self, args: argparse.Namespace):
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

        self.data_dir = self.args.data_dir
        self.tag = self.args.tag
        input_size = 0
        if self.tag == 'fasttext':
            input_size = 100
        elif self.tag == 'tfidf':
            input_size = 13047
        elif self.tag == 'bionerds':
            input_size = 7
        self.model = MLPModel( input_size, args.hidden_size, args.dropout_rate, args.act_func)
        logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
        self.result_logger = logging.getLogger(__name__)
        self.result_logger.setLevel(logging.INFO)
        self.result_logger.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))
        self.bce_loss = BCEWithLogitsLoss(reduction="mean")

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
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)
        t_total = (len(self.train_dataloader()) // self.args.accumulate_grad_batches + 1) * self.args.max_epochs
        if self.args.lr_mini == -1:
            lr_mini = self.args.lr / 5
        else:
            lr_mini = self.args.lr_mini
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, self.args.warmup_steps, t_total, lr_end=lr_mini)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids):
        return self.model(input_ids)

    def compute_loss(self, preds, trues):
        single_loss = self.bce_loss(preds.float(), trues.float())
        return single_loss

    def training_step(self, batch, batch_idx):
        tf_board_logs = {
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }
        X_train, Y_train = batch
        single_logits = self(X_train)
        total_loss = self.compute_loss(preds=single_logits, trues=Y_train)
        tf_board_logs[f"train_loss"] = total_loss
        return {'loss': total_loss, 'log': tf_board_logs}

    def span_f1(self, preds, trues):
        gt = trues.bool().view(-1)
        logits = (preds > 0).view(-1)
        tp = ( logits & gt ).long().sum()
        fp = ( logits &~gt ).long().sum()
        fn = (~logits & gt ).long().sum()
        tn = (~logits &~gt ).long().sum()
        return torch.stack([tp, fp, fn])

    def validation_step(self, batch, batch_idx):
        output = {}
        X_train, Y_train = batch
        single_logits = self(X_train)

        total_loss = self.compute_loss(preds=single_logits, trues=Y_train)
        output[f"val_loss"] = total_loss
        single_preds = single_logits > 0
        span_f1_stats = self.span_f1(preds=single_preds, trues=Y_train)
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
        X_train, Y_train = batch
        single_logits = self(X_train)
        single_preds = single_logits > 0
        span_f1_stats = self.span_f1(preds=single_preds, trues=Y_train)
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
        return self.get_dataloader("train", self.tag)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("dev", self.tag)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", self.tag)

    def get_dataloader(self, prefix, tag) -> DataLoader:
        if tag == 'fasttext':
            fname = os.path.join(self.data_dir, f"{prefix}.pkl")
            dataset = fasttextDataset(fname=fname)
            dataloader = DataLoader(dataset=dataset, batch_size=self.args.batch_size, num_workers=self.args.workers, shuffle=True if prefix == "train" else False, collate_fn=collate)
        elif tag == 'tfidf':
            fname = os.path.join(self.data_dir, f"{prefix}.pkl")
            dataset = tfidfDataset(fname=fname)
            dataloader = DataLoader(dataset=dataset, batch_size=self.args.batch_size, num_workers=self.args.workers, shuffle=True if prefix == "train" else False, collate_fn=collate)
        elif tag == 'bionerds':
            fname = os.path.join(self.data_dir, f"{prefix}.csv")
            dataset = bioNerDSDataset(fname=fname)
            dataloader = DataLoader(dataset=dataset, batch_size=self.args.batch_size, num_workers=self.args.workers, shuffle=True if prefix == "train" else False)
        return dataloader

def find_best_checkpoint_on_dev(output_dir: str, log_file: str = "eval_result_log.txt", only_keep_the_best_ckpt: bool = False):
    with open(os.path.join(output_dir, log_file)) as f:
        log_lines = f.readlines()

    F1_PATTERN = re.compile(r"span_f1 reached \d+\.\d* \(best")
    # val_f1 reached 0.00000 (best 0.00000)
    CKPT_PATTERN = re.compile(r"saving model to \S+ as top")
    checkpoint_info_lines = []
    for log_line in log_lines:
        if "saving model to" in log_line:
            checkpoint_info_lines.append(log_line)
    # example of log line
    # Epoch 00000: val_f1 reached 0.00000 (best 0.00000), saving model to /data/xiaoya/outputs/0117/debug_5_12_2e-5_0.001_0.001_275_0.1_1_0.25/checkpoint/epoch=0.ckpt as top 20
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

# LR = "1e-5"
# data = "bioNerDS"
# tag = "bionerds"
# BASE = "/home/b317l704"
# REPO_PATH = f"{BASE}/sentence_classifer_git"
# DATA_DIR = f"{REPO_PATH}/datasets/{data}"
# OUTPUT_BASE = f"{REPO_PATH}/outputs"
# BATCH = "4"
# GRAD_ACC = "4"
# DROPOUT = "0.1"
# LR_MINI = "3e-8"
# LR_SCHEDULER = "polydecay"
# SPAN_WEIGHT = "0.1"
# WARMUP = "0"
# MAX_NORM = "1.0"
# MAX_EPOCH = "10"
# INTER_HIDDEN = "2048"
# WEIGHT_DECAY = "0.01"
# OPTIM = "torch.adam"
# VAL_CHECK = "0.2"
# PREC = "16"
# WORKERS = "6"
# OUTPUT_DIR = f"{OUTPUT_BASE}/{data}/lr{LR}"
#
# argv = [
# "--data_dir", f"{DATA_DIR}",
# "--default_root_dir", f"{OUTPUT_DIR}",
# "--batch_size", f"{BATCH}",
# "--lr", f"{LR}",
# "--workers", f"{WORKERS}",
# "--weight_decay", f"{WEIGHT_DECAY}",
# "--warmup_steps", f"{WARMUP}",
# "--seed", "0",
# "--tag", f"{tag}",
# "--hidden_size", f"{INTER_HIDDEN}",
# "--dropout_rate", f"{DROPOUT}",
# "--optimizer", f"{OPTIM}",
# "--lr_scheduler", f"{LR_SCHEDULER}",
# "--lr_mini", f"{LR_MINI}",
# "--gpus", "1",
# "--precision", f"{PREC}",
# "--progress_bar_refresh_rate", "1",
# "--val_check_interval", f"{VAL_CHECK}",
# "--accumulate_grad_batches", f"{GRAD_ACC}",
# "--max_epochs", f"{MAX_EPOCH}",
# "--distributed_backend", "ddp",
# "--gradient_clip_val", f"{MAX_NORM}"]

def main():
    """main"""
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    # args = parser.parse_args(argv)

    set_random_seed(args.seed)
    model = SimpleModel(args)
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
