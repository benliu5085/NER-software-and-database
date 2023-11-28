import os
import torch
import argparse
import pickle
from torch.utils.data import DataLoader
from utils.random_seed import set_random_seed
set_random_seed(0)
from mlp.dataset import fasttextDataset, tfidfDataset, bioNerDSDataset, collate
from mlp.trainer import SimpleModel

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

def get_dataloader(config, prefix) -> DataLoader:
    if config.tag == 'fasttext':
        fname = os.path.join(config.data_dir, f"{prefix}.pkl")
        dataset = fasttextDataset(fname=fname)
        dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=1, shuffle=False, collate_fn=collate)
    elif config.tag == 'tfidf':
        fname = os.path.join(config.data_dir, f"{prefix}.pkl")
        dataset = tfidfDataset(fname=fname)
        dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=1, shuffle=False, collate_fn=collate)
    elif config.tag == 'bionerds':
        fname = os.path.join(config.data_dir, f"{prefix}.csv")
        dataset = bioNerDSDataset(fname=fname)
        dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=1, shuffle=False)
    return dataloader


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="inference the model output.")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--model_ckpt", type=str, default="")
    parser.add_argument("--hparams_file", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--tag", choices=["tfidf", "fasttext", "bionerds"], default="tfidf")
    return parser

# argv = [
# "--data_dir", "datasets/fasttext",
# "--model_ckpt", "outputs/fasttext/lr5e-2/epoch=2.ckpt",
# "--hparams_file", "outputs/fasttext/lr5e-2/lightning_logs/version_0/hparams.yaml",
# "--output_dir",   "outputs/fasttext/lr5e-2",
# "--tag", "fasttext"
# ]
def main():
    parser = get_parser()
    args = parser.parse_args()
    trained_model = SimpleModel.load_from_checkpoint(
        checkpoint_path=args.model_ckpt,
        hparams_file=args.hparams_file,
        map_location=None,
        batch_size=1,
        workers=1)
    pick_list = []
    list_path = os.path.join(args.data_dir, "example.txt")
    with open(list_path, 'r') as fin:
        for l in fin:
            pick_list.append(l.strip())
    data = {}
    for pmcid in pick_list:
        print(f"predict {pmcid} ...")
        data[pmcid] = []
        data_loader = get_dataloader(args, pmcid)
        for batch in data_loader:
            X_train, Y_train = batch
            preds = trained_model.model(X_train)
            data[pmcid].append(preds.squeeze().detach().numpy())
    pickle.dump(data, open(f"{args.output_dir}/{args.tag}.DS0.pkl", 'wb'))

if __name__ == "__main__":
    main()
