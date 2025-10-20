
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

def get_scores(predictions, references):
    clean_data = [
        (pred.strip().replace("\n", ""), ref.strip().replace("\n", ""))
        for pred, ref in zip(predictions, references)
    ]

    clean_preds, clean_refs = zip(*clean_data)

    clean_refs_list = [[ref] for ref in clean_refs]

    def compute_bleu_batch(preds, refs):
        scores = []
        for p, r in zip(preds, refs):
            score = sacrebleu.sentence_bleu(p, [r]).score
            scores.append(score)
        mean_score = sum(scores) / len(scores)
        return mean_score / 100

    bleu = compute_bleu_batch(clean_preds, clean_refs)

    # rouge
    r1_total, r2_total, rl_total = 0, 0, 0
    for pred, refs in zip(clean_preds, clean_refs_list):
        ref = refs[0]
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer = True)
        scores = rouge.score(ref, pred)
        r1_total += scores["rouge1"].fmeasure
        r2_total += scores["rouge2"].fmeasure
        rl_total += scores["rougeL"].fmeasure
    n = len(clean_preds)
    rouge1 = r1_total / n
    rouge2 = r2_total / n
    rougeL = rl_total / n

    # meteor
    meteor_total = 0
    for pred, refs in zip(clean_preds, clean_refs_list):
        meteor_total += meteor_score(
            [ref.split() for ref in refs],
            pred.split()
        )
    meteor = meteor_total / n
    
    # chrf++
    chrf_scorer = sacrebleu.metrics.CHRF(char_order = 6, word_order = 2, beta = 2)
    chrf = chrf_scorer.corpus_score(predictions, [references]).score / 100

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
    res = {
        "bleu": [0],
        "rouge1": [0],
        "rouge2": [0],
        "rougeL": [0],
        "meteor": [0],
        "chrf++": [0],
    }
    for ref in refs:
        scores = get_scores([pred], [ref])
        for k, v in scores.items():
            res[k].append(v)
            
    for k, v in res.items():
        res[k] = np.max(v)
    
    return res

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
        