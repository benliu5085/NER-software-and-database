import os
import sys

def parse(fname):
    F1 = 0
    ckpt = ""
    with open(fname) as fin:
        for l in fin:
            if l.find("Best F1") > 0:
                F1 = float(l.split()[-1])
            elif l.find("Best checkpoint on DEV set") > 0:
                ckpt = l[l.find('outputs/')+8:-1]
    return (F1, ckpt)

def findBestCheckPoint(fdir):
    BF, BCKPT = 0, ""
    for subfdir in os.listdir(fdir):
        this_f1, this_ckpt = parse(f"{fdir}/{subfdir}/eval_result_log.txt")
        if this_f1 > BF:
            BF, BCKPT = this_f1, this_ckpt
    return BCKPT

helper =  "call by ~ <mode> <rep1> <rep2> ..."
helper += "    <mode> - running mode, choose from {sci[1-4], bio[1-4]}"
helper += "             sci1: retrain using sciBERT  (b0, b1,   .. b4  );"
helper += "             sci2: pretrain using sciBERT (b0, b+1,  .. b+4 );"
helper += "             sci3: pretrain using sciBERT (b0, b+1_, .. b+4_);"
helper += "             sci4: pretrain using sciBERT (b0, b1_,  .. b4_);"
helper += "             bio[1-4]: same setting but using bioBERT;"
helper += "    <rep1> <rep2> ..."
helper += "           - tell which rep to run, must be integer {0, .., 9}."

if len(sys.argv) > 2:
    mode_code = sys.argv[1]
    rep = sys.argv[2:]
else:
    print(helper)
    exit()

seed = [0, 2023, 604641129, 449958349, 196965608, 127718410, 500646678, 772810659, 881783626, 948751968]
map_running = {
    'sci1': ["tuning_maskedNER",          104, "scibert_scivocab_uncased"],
    'sci2': ["tuning_pretrain_maskedNER", 104, "scibert_scivocab_uncased"],
    'sci3': ["tuning_pretrain_maskedNER", 104, "scibert_scivocab_uncased"],
    'sci4': ["tuning_pretrain_maskedNER", 104, "scibert_scivocab_uncased"],
    'bio1': ["tuning_maskedNER",          103, "biobert_v1.1_pubmed"],
    'bio2': ["tuning_pretrain_maskedNER", 103, "biobert_v1.1_pubmed"],
    'bio3': ["tuning_pretrain_maskedNER", 103, "biobert_v1.1_pubmed"],
    'bio4': ["tuning_pretrain_maskedNER", 103, "biobert_v1.1_pubmed"],
}

map_training = {
    'sci1': ["pmc60+_v0_g2_b1", "pmc60+_v0_g2_b2", "pmc60+_v0_g2_b3", "pmc60+_v0_g2_b4"],
    'bio1': ["pmc60+_v0_g2_b1", "pmc60+_v0_g2_b2", "pmc60+_v0_g2_b3", "pmc60+_v0_g2_b4"],
    'sci2': ["pmc60+_v0_g2_b+1", "pmc60+_v0_g2_b+2", "pmc60+_v0_g2_b+3", "pmc60+_v0_g2_b+4"],
    'bio2': ["pmc60+_v0_g2_b+1", "pmc60+_v0_g2_b+2", "pmc60+_v0_g2_b+3", "pmc60+_v0_g2_b+4"],
    'sci3': ["pmc60+_v0_g2_b+1_", "pmc60+_v0_g2_b+2_", "pmc60+_v0_g2_b+3_", "pmc60+_v0_g2_b+4_"],
    'bio3': ["pmc60+_v0_g2_b+1_", "pmc60+_v0_g2_b+2_", "pmc60+_v0_g2_b+3_", "pmc60+_v0_g2_b+4_"],
    'sci4': ["pmc60+_v0_g2_b1_", "pmc60+_v0_g2_b2_", "pmc60+_v0_g2_b3_", "pmc60+_v0_g2_b4_"],
    'bio4': ["pmc60+_v0_g2_b1_", "pmc60+_v0_g2_b2_", "pmc60+_v0_g2_b3_", "pmc60+_v0_g2_b4_"],
}

if mode_code not in map_running:
    print(helper)
    print("\nError: you must choose <mode> from {sci[1-4], bio[1-4]}")
    exit()

for i, v in enumerate(rep):
    try:
        rep[i] = int(v)
    except:
        print(helper)
        print(f"\nError: <rep{i+1}> is not an integer!")
        exit()

if len(set(rep)-set(range(10))) > 0:
    print(helper)
    print("\nError: <rep[1-10]> must be integer {0, .. ,9}")
    exit()

base = "pmc60+_v0_g2_b0"
datasets = map_training[mode_code]
params = map_running[mode_code]
grid_search = ["1e-7", "5e-7", "1e-6", "5e-6", "1e-5", "5e-5"]
which_bert = f"{mode_code[:3]}BERT"
if mode_code in ['sci1', 'bio1']:
    # bash ~.sh <DS> <REP> <LR> <SEED> <MASK_ID> <WHICH_BERT> <FILE>
    for this_rep in rep:
        for ds in [base] + datasets:
            for lr in grid_search:
                cmd = f"bash maskedNER/{params[0]}.sh {ds} REP{this_rep+1} {lr} {seed[this_rep]} {params[1]} {params[2]} {which_bert}"
                os.system(cmd)
elif mode_code[:3] == 'sci':
    for this_rep in rep:
        if os.path.exists(f"outputs/REP{this_rep+1}/pmc60+_v0_g2_b0") and len(os.listdir(f"outputs/REP{this_rep+1}/pmc60+_v0_g2_b0")) == 6:
            # bash ~.sh <DS> <REP> <LR> <SEED> <MASK_ID> <WHICH_BERT> <FILE> <CKPT>
            best_ckpt = findBestCheckPoint(f"outputs/REP{this_rep+1}/pmc60+_v0_g2_b0/{which_bert}")
            for ds in datasets:
                for lr in grid_search:
                    cmd = f"bash maskedNER/{params[0]}.sh {ds} REP{this_rep+1} {lr} {seed[this_rep]} {params[1]} {params[2]} {which_bert} {best_ckpt}"
                    os.system(cmd)
                best_ckpt = findBestCheckPoint(f"outputs/REP{this_rep+1}/{ds}/{which_bert}")
        else:
            print("You should run retrain before run pretrain")
            exit()
