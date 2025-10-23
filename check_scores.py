
import sys
from sacrebleu import corpus_bleu
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import *

results_reader = ModelUtils.TestLogger.ResultsReader(sys.argv[1])
df = pd.DataFrame({
    "prediction": results_reader.predictions,
    "label": results_reader.labels
})
df["original"] = pd.read_csv("data/test.csv")["original"]

def get_scores(pred, refs):
    clean_pred = pred.strip().replace("\n", "")
    clean_refs = [ref.strip().replace("\n", "") for ref in refs]

    # bleu
    bleu = sacrebleu.sentence_bleu(clean_pred, clean_refs).score / 100

    # rouge
    r1_total, r2_total, rl_total = 0, 0, 0
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer = True)
    scores = rouge.score(clean_refs, clean_pred)
    rouge1 = scores["rouge1"].fmeasure
    rouge2 = scores["rouge2"].fmeasure
    rougeL = scores["rougeL"].fmeasure

    # meteor
    meteor = meteor_score(
        [ref.split() for ref in refs],
        pred.split()
    )
    
    # chrf++
    chrf_scorer = sacrebleu.metrics.CHRF(char_order = 6, word_order = 2, beta = 2)
    chrf = chrf_scorer.sentence_score(clean_pred, clean_refs).score

    return {
        "bleu": bleu,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "meteor": meteor,
        "chrf++": chrf,
    }
    
def get_res(i):
    refs = df[df["original"] == df["original"][i]]["label"].tolist()
    pred = df["prediction"][i]
    scores = get_scores(pred, refs)    
    return scores

res = {
    "bleu": [0],
    "rouge1": [0],
    "rouge2": [0],
    "rougeL": [0],
    "meteor": [0],
    "chrf++": [0],
}
pbar = tqdm(range(len(df)))
for i in pbar:
    scores = get_res(i)
    for k, v in scores.items():
        res[k].append(v)
    pbar.set_postfix({
        k: round(np.mean(v), 5) for k, v in res.items()
    })
    
for k, v in res.items():
    print(k, np.mean(v))

joblib.dump(res, "check_scores.results")
        
