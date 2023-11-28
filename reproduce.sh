### change this to your root directory
BASE=/home/b317l704/sentence_classifer_git
cd ${BASE}
### assuming you use conda to manage the python virtual environment
conda activate torch1.7.1_CUDA11

"""========== BERT+mask =========="""
### make dataset
cd ${BASE}
# bio1 / sci1
python ner2maskedMRC.py pmc60+_v0_g2_b0
python ner2maskedMRC.py pmc60+_v0_g2_b1
python ner2maskedMRC.py pmc60+_v0_g2_b2
python ner2maskedMRC.py pmc60+_v0_g2_b3
python ner2maskedMRC.py pmc60+_v0_g2_b4
# bio4 / sci4
python ner2maskedMRC.py pmc60+_v0_g2_b1_
python ner2maskedMRC.py pmc60+_v0_g2_b2_
python ner2maskedMRC.py pmc60+_v0_g2_b3_
python ner2maskedMRC.py pmc60+_v0_g2_b4_

### run training
# REP1 ${lr} 0
# REP2 ${lr} 2023
# REP3 ${lr} 604641129
# REP4 ${lr} 449958349
# REP5 ${lr} 196965608

## run_masked_NER.py contains all hyperparameters
cd ${BASE}
python run_masked_NER.py sci1 0 1 2 3 4
python run_masked_NER.py bio1 0 1 2 3 4
python run_masked_NER.py sci4 0 1 2 3 4
python run_masked_NER.py bio4 0 1 2 3 4
## or you can just call
# bash maskedNER/tuning_maskedNER.sh       <DS> <REP> <LR> <SEED> <MASK_ID> <WHICH_BERT> <FILE>
# bash maskedNER/tuning_pretrain_maskedNER <DS> <REP> <LR> <SEED> <MASK_ID> <WHICH_BERT> <FILE> <CKPT>
## change other hyperparameters from maskedNER/tuning_maskedNER.sh or maskedNER/tuning_pretrain_maskedNER.sh

### Testing
## test on test1, test2, test1+, test2+
for ACC in "REP1" "REP2" "REP3" "REP4" "REP5";
  do
  for ds in "pmc60+_v0_g2_b0"  "pmc60+_v0_g2_b1"  "pmc60+_v0_g2_b2"  "pmc60+_v0_g2_b3"  "pmc60+_v0_g2_b4"  "pmc60+_v0_g2_b1_"  "pmc60+_v0_g2_b2_"  "pmc60+_v0_g2_b3_"  "pmc60+_v0_g2_b4_";
    do
    python masked_NER_predict.py $ACC $ds bio;
    python masked_NER_predict.py $ACC $ds sci;
  done
done
## full paper test
for ACC in "REP1" "REP2" "REP3" "REP4" "REP5";
  do
  for ds in "pmc60+_v0_g2_b0"  "pmc60+_v0_g2_b1"  "pmc60+_v0_g2_b2"  "pmc60+_v0_g2_b3"  "pmc60+_v0_g2_b4"  "pmc60+_v0_g2_b1_"  "pmc60+_v0_g2_b2_"  "pmc60+_v0_g2_b3_"  "pmc60+_v0_g2_b4_";
    do
    python masked_NER_fullTest.py $ACC $ds bio;
    python masked_NER_fullTest.py $ACC $ds sci;
  done
done
## or you can call
# bash maskedNER/predict.sh <DS> <TestSet> <OUTDIR> <CKPT> <MASK_ID> <WHICH_BERT>

### find the predicted TSV file from folders inside outputs.
### The TSV file contains the prediction score of each instance.
### You can use the sigmoid to transform the prediction score into probability.

"""========== BERT + unmask =========="""
### make dataset
cd ${BASE}
python ner2unmaskedMRC.py

### run training
cd ${BASE}
for lr in "1e-7" "5e-7" "1e-6" "5e-6" "1e-5" "5e-5";
do
  bash unmaskedNER/tuning_unmaskedNER.sh $lr scibert_scivocab_uncased sciBERT;
  bash unmaskedNER/tuning_unmaskedNER.sh $lr biobert_v1.1_pubmed bioBERT;
done

### You should find the checkpoints from outputs for prediction
### unmasked BERT was trained on DS0
# bash unmaskedNER/predict.sh <OUTDIR> <CKPT> <WHICH_BERT>
bash unmaskedNER/predict.sh outputs/DS0/bioBERT/lr1e-5_maxlen200 epoch=7_v0.ckpt biobert_v1.1_pubmed
bash unmaskedNER/predict.sh outputs/DS0/bioBERT/lr1e-6_maxlen200 epoch=9_v0.ckpt biobert_v1.1_pubmed
bash unmaskedNER/predict.sh outputs/DS0/bioBERT/lr1e-7_maxlen200 epoch=0_v0.ckpt biobert_v1.1_pubmed
bash unmaskedNER/predict.sh outputs/DS0/bioBERT/lr5e-5_maxlen200 epoch=7.ckpt    biobert_v1.1_pubmed
bash unmaskedNER/predict.sh outputs/DS0/bioBERT/lr5e-6_maxlen200 epoch=8.ckpt    biobert_v1.1_pubmed
bash unmaskedNER/predict.sh outputs/DS0/bioBERT/lr5e-7_maxlen200 epoch=9_v1.ckpt biobert_v1.1_pubmed
bash unmaskedNER/predict.sh outputs/DS0/sciBERT/lr1e-5_maxlen200 epoch=6.ckpt    scibert_scivocab_uncased
bash unmaskedNER/predict.sh outputs/DS0/sciBERT/lr1e-6_maxlen200 epoch=8.ckpt    scibert_scivocab_uncased
bash unmaskedNER/predict.sh outputs/DS0/sciBERT/lr1e-7_maxlen200 epoch=0.ckpt    scibert_scivocab_uncased
bash unmaskedNER/predict.sh outputs/DS0/sciBERT/lr5e-5_maxlen200 epoch=5_v0.ckpt scibert_scivocab_uncased
bash unmaskedNER/predict.sh outputs/DS0/sciBERT/lr5e-6_maxlen200 epoch=8.ckpt    scibert_scivocab_uncased
bash unmaskedNER/predict.sh outputs/DS0/sciBERT/lr5e-7_maxlen200 epoch=9_v0.ckpt scibert_scivocab_uncased

"""========== TF-IDF/fasttext/bioNerDS + NN =========="""
### make dataset
cd ${BASE}/ner2mrc
python ner2TFIDF.py
python ner2fasttext.py
## We provided the processed bioNerDS feature

### run training
### they takes several mins to train and requires larger learning rate
cd ${BASE}
for lr in "1e-7" "5e-7" "1e-6" "5e-6" "1e-5" "5e-5" "1e-4" "5e-4" "1e-3" "5e-3" "1e-2" "5e-2" "1e-1" "5e-1";
do
  bash mlp/my_tuningMLP.sh $lr bioNerDS bionerds 100; # change hidden units to 100
  bash mlp/my_tuningMLP.sh $lr TFIDF tfidf;
  bash mlp/my_tuningMLP.sh $lr fasttext fasttext;
done

### inference
### You should find the checkpoints from outputs for prediction
### They have their own directories
# bash mlp/predict.sh <DATADIR> <OUTDIR> <CKPT> <WHICH_Feature>
bash mlp/predict.sh datasets/fasttext outputs/fasttext/lr1e-1 epoch=6_v1.ckpt fasttext
bash mlp/predict.sh datasets/fasttext outputs/fasttext/lr1e-2 epoch=5.ckpt fasttext
bash mlp/predict.sh datasets/fasttext outputs/fasttext/lr1e-3 epoch=8_v0.ckpt fasttext
bash mlp/predict.sh datasets/fasttext outputs/fasttext/lr1e-4 epoch=5.ckpt fasttext
bash mlp/predict.sh datasets/fasttext outputs/fasttext/lr1e-5 epoch=0_v4.ckpt fasttext
bash mlp/predict.sh datasets/fasttext outputs/fasttext/lr1e-6 epoch=0_v1.ckpt fasttext
bash mlp/predict.sh datasets/fasttext outputs/fasttext/lr1e-7 epoch=9.ckpt fasttext
bash mlp/predict.sh datasets/fasttext outputs/fasttext/lr5e-1 epoch=9_v0.ckpt fasttext
bash mlp/predict.sh datasets/fasttext outputs/fasttext/lr5e-2 epoch=2.ckpt fasttext
bash mlp/predict.sh datasets/fasttext outputs/fasttext/lr5e-3 epoch=1.ckpt fasttext
bash mlp/predict.sh datasets/fasttext outputs/fasttext/lr5e-4 epoch=8_v0.ckpt fasttext
bash mlp/predict.sh datasets/fasttext outputs/fasttext/lr5e-5 epoch=0_v1.ckpt fasttext
bash mlp/predict.sh datasets/fasttext outputs/fasttext/lr5e-6 epoch=0.ckpt fasttext
bash mlp/predict.sh datasets/fasttext outputs/fasttext/lr5e-7 epoch=1.ckpt fasttext

bash mlp/predict.sh datasets/TFIDF outputs/TFIDF/lr1e-1 epoch=2.ckpt tfidf
bash mlp/predict.sh datasets/TFIDF outputs/TFIDF/lr1e-2 epoch=1_v0.ckpt tfidf
bash mlp/predict.sh datasets/TFIDF outputs/TFIDF/lr1e-3 epoch=2_v1.ckpt tfidf
bash mlp/predict.sh datasets/TFIDF outputs/TFIDF/lr1e-4 epoch=1.ckpt tfidf
bash mlp/predict.sh datasets/TFIDF outputs/TFIDF/lr1e-5 epoch=0_v4.ckpt tfidf
bash mlp/predict.sh datasets/TFIDF outputs/TFIDF/lr1e-6 epoch=1_v0.ckpt tfidf
bash mlp/predict.sh datasets/TFIDF outputs/TFIDF/lr1e-7 epoch=9.ckpt tfidf
bash mlp/predict.sh datasets/TFIDF outputs/TFIDF/lr5e-1 epoch=0.ckpt tfidf
bash mlp/predict.sh datasets/TFIDF outputs/TFIDF/lr5e-2 epoch=1_v0.ckpt tfidf
bash mlp/predict.sh datasets/TFIDF outputs/TFIDF/lr5e-3 epoch=1_v0.ckpt tfidf
bash mlp/predict.sh datasets/TFIDF outputs/TFIDF/lr5e-4 epoch=0_v0.ckpt tfidf
bash mlp/predict.sh datasets/TFIDF outputs/TFIDF/lr5e-5 epoch=3.ckpt tfidf
bash mlp/predict.sh datasets/TFIDF outputs/TFIDF/lr5e-6 epoch=0.ckpt tfidf
bash mlp/predict.sh datasets/TFIDF outputs/TFIDF/lr5e-7 epoch=8.ckpt tfidf

bash mlp/predict.sh datasets/bioNerDS outputs/bioNerDS/lr1e-1 epoch=3.ckpt bionerds
bash mlp/predict.sh datasets/bioNerDS outputs/bioNerDS/lr1e-2 epoch=1.ckpt bionerds
bash mlp/predict.sh datasets/bioNerDS outputs/bioNerDS/lr1e-3 epoch=6.ckpt bionerds
bash mlp/predict.sh datasets/bioNerDS outputs/bioNerDS/lr1e-4 epoch=0_v0.ckpt bionerds
bash mlp/predict.sh datasets/bioNerDS outputs/bioNerDS/lr1e-5 epoch=0_v4.ckpt bionerds
bash mlp/predict.sh datasets/bioNerDS outputs/bioNerDS/lr1e-6 epoch=0_v0.ckpt bionerds
bash mlp/predict.sh datasets/bioNerDS outputs/bioNerDS/lr1e-7 epoch=0.ckpt bionerds
bash mlp/predict.sh datasets/bioNerDS outputs/bioNerDS/lr5e-1 epoch=1.ckpt bionerds
bash mlp/predict.sh datasets/bioNerDS outputs/bioNerDS/lr5e-2 epoch=3_v0.ckpt bionerds
bash mlp/predict.sh datasets/bioNerDS outputs/bioNerDS/lr5e-3 epoch=1.ckpt bionerds
bash mlp/predict.sh datasets/bioNerDS outputs/bioNerDS/lr5e-4 epoch=6.ckpt bionerds
bash mlp/predict.sh datasets/bioNerDS outputs/bioNerDS/lr5e-5 epoch=0_v0.ckpt bionerds
bash mlp/predict.sh datasets/bioNerDS outputs/bioNerDS/lr5e-6 epoch=0.ckpt bionerds
bash mlp/predict.sh datasets/bioNerDS outputs/bioNerDS/lr5e-7 epoch=0_v2.ckpt bionerds
