import os
import sys

def parse(fname):
    ckpt = ""
    with open(fname) as fin:
        for l in fin:
            if l.find("Best checkpoint on DEV set") > 0:
                ckpt = l[l.find('epoch'):-1]
    return (ckpt)

def run_predictor():
    helper =  "call by ~ <ACC> <DS> <BERT> ..."
    helper += "    <ACC>  - Repeat code, choose from {REP1, REP2, ..., REP10}"
    helper += "    <DS>   - training dataset"
    helper += "    <BERT> - which bert used, choose from {sci, bio}"
    if len(sys.argv) == 4 :
        ACC  = sys.argv[1]
        DS   = sys.argv[2]
        BERT = sys.argv[3]
    else:
        print(helper)
        exit()

    if ACC not in [f"REP{x}" for x in range(1,11)]:
        print(helper)
        print("\nError: you must choose <ACC> from {REP1, REP1, ..., REP10}")
        exit()

    map_bert = { 'sci': "104 scibert_scivocab_uncased",
                 'bio': "103 biobert_v1.1_pubmed"}

    if BERT not in map_bert:
        print(helper)
        print("\nError: you must choose <BERT> from {sci, bio}")
        exit()

    OUT_BASE = f"outputs/{ACC}/{DS}/{BERT}BERT" # outputs/REP1/pmc60+_v0_g2_b0/sciBERT
    if not os.path.exists(OUT_BASE):
        print(f"Error: directory {OUT_BASE} doesn't exist, did you train the model?")
        exit()

    for subfdir in os.listdir(OUT_BASE):
        for ts in [DS, "pmc60+_g2", "pmc60_g2"]:
            outdir = f"{OUT_BASE}/{subfdir}" # outputs/REP1/pmc60+_v0_g2_b0/sciBERT/lrxxx
            ckpt = parse(f"{outdir}/eval_result_log.txt")
            cmd = f"bash maskedNER/predict.sh {DS} {ts} {outdir} {ckpt} {map_bert[BERT]}"
            print(cmd)
            os.system(cmd)

if __name__ == '__main__':
    run_predictor()
