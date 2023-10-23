import json
import torch
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset

def collate_to_max_length(batch):
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens,
            token_type_ids,
            sample_idx, label_idx,
            single_labels
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    for field_idx in range(2):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    output.append(torch.stack([x[2] for x in batch]))
    output.append(torch.stack([x[3] for x in batch]))

    field_idx = 4
    pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][field_idx]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)

    return output

class MRCNERDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        pad_to_maxlen: bool
    """
    def __init__(self, json_path, tokenizer, max_length=512, pad_to_maxlen=False):
        self.all_data = json.load(open(json_path, encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_maxlen = pad_to_maxlen

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            sample_idx: sample id
            label_idx: label id
            single_labels: labels of NER in tokens, [seq_len]
        """
        data = self.all_data[item]
        tokenizer = self.tokenizer

        qas_id = data.get("qas_id", "0.0")
        sample_idx, label_idx = qas_id.split(".")
        sample_idx = torch.LongTensor([int(sample_idx)])
        label_idx = torch.LongTensor([int(label_idx)])

        query = data["query"]
        context = data["context"]
        start_positions = data["start_position"]
        end_positions = data["end_position"]

        words = context.split()
        start_positions = [x + sum([len(w) for w in words[:x]]) for x in start_positions]
        end_positions = [x + sum([len(w) for w in words[:x + 1]]) for x in end_positions]

        query_context_tokens = tokenizer.encode(query, context, add_special_tokens=True)
        tokens = query_context_tokens.ids
        type_ids = query_context_tokens.type_ids
        offsets = query_context_tokens.offsets

        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. word-piece tokenize
        origin_offset2token_idx_start = {}
        origin_offset2token_idx_end = {}
        for token_idx in range(len(tokens)):
            # skip query tokens
            if type_ids[token_idx] == 0:
                continue
            token_start, token_end = offsets[token_idx]
            # skip [CLS] or [SEP]
            if token_start == token_end == 0:
                continue
            origin_offset2token_idx_start[token_start] = token_idx
            origin_offset2token_idx_end[token_end] = token_idx

        new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions]
        new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]
        assert len(new_start_positions) == len(new_end_positions) == len(start_positions)

        # truncate
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]

        # make sure last token is [SEP]
        sep_token = tokenizer.token_to_id("[SEP]")
        if tokens[-1] != sep_token:
            assert len(tokens) == self.max_length
            tokens = tokens[: -1] + [sep_token]

        if self.pad_to_maxlen:
            tokens = self.pad(tokens, 0)
            type_ids = self.pad(type_ids, 1)

        seq_len = len(tokens)

        single_labels = torch.zeros([seq_len], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            for tmpi in range(start, end+1):
                single_labels[tmpi] = 1
        return [
            torch.LongTensor(tokens),
            torch.LongTensor(type_ids),
            sample_idx,
            label_idx,
            torch.LongTensor(single_labels), # for sentence classifier purpose
        ]

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst


def run_dataset():
    """test dataset"""
    import os
    from torch.utils.data import DataLoader


    bert_path = "/home/b317l704/scibert/preTrain/tensorflow"
    vocab_file = os.path.join(bert_path, "vocab.txt")
    json_path = "/home/b317l704/sentence_classifer_git/datasets/pmc60+_v0_g2_b0/mrc-ner.train"

    tokenizer = BertWordPieceTokenizer(vocab_file)

    dataset = MRCNERDataset(json_path=json_path,
                            tokenizer=tokenizer,
                            max_length=200,
                            pad_to_maxlen=False)
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, collate_fn=collate_to_max_length)

    # bert_config = BertQueryNerConfig.from_pretrained(
    #     bert_path,
    #     hidden_dropout_prob=0.1,
    #     attention_probs_dropout_prob=0.1,
    #     mrc_dropout=0.3,
    #     classifier_act_func = 'gelu',
    #     classifier_intermediate_hidden_size=2048)
    # model = BertQueryNER.from_pretrained(bert_path, config=bert_config)
    for batch in dataloader:
        tokens, token_type_ids, sample_idx, label_idx, single_labels = batch
        print(tokens.shape, single_labels.shape, single_labels.sum())
        # attention_mask = (tokens != 0).long()
        # single_logits = model(tokens, attention_mask, token_type_ids)
        # print(single_logits.shape)

if __name__ == '__main__':
    run_dataset()
