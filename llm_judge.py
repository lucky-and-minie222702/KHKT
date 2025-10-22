import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *
import re
import sys
import pandas as pd
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

config = load_json(sys.argv[1])

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code = True,
    dtype = torch.bfloat16,
    device_map = "auto",
    # low_cpu_mem_usage = True,
)
tokenizer.padding_side = "left"


def build_adjudicator_prompt(question, model_response, complexity, question_class, original, answer):
    # Ensure iterable
    question_class = list(question_class) if isinstance(question_class, (list, tuple)) else [str(question_class)]
    aspects_json = ",\n".join([
        f'    "{aspect}": {{\n      "score": 0 or 1,\n      "reason": "<short justification>"\n    }}'
        for aspect in question_class
    ])
    return f""" 
## CONTEXT
The current **exam question** is derived from one or more original Q/A items (see `original`) and may vary in complexity. It has been annotated with one or more **aspect labels** (see `question_class`), where each label represents a specific area of clinical knowledge (e.g., diagnosis, treatment, anatomy, procedures, interpretation, etc.).

---

## TASK
Your job is to **grade the doctor’s response** against each individual aspect.

For **each aspect** in `question_class`, you must:
- Compare the **doctor's response** to the **correct answer**.
- Use the **original Q/A pairs** for supporting context.
- Assign:
  - `\\"score\\": 1` if the response fully and correctly addresses that aspect.
  - `\\"score\\": 0` if the response is partially correct, incorrect, missing, or misinterprets that aspect.
- Give a **short reason** for the score.

---

## OUTPUT FORMAT (STRICT)
Return a **valid JSON object** where:
- Each key is one aspect label from `question_class`.
- Each value is a dictionary with `\\"score\\"` and `\\"reason\\"`.
- The entire JSON must be wrapped in triple backticks with `json` as the language identifier.
- No extra comments, text, or output outside the code block.

**Template:**
```json
{{
{aspects_json}
}}
Input:
Exam Question: {question}
Doctor’s Response: {model_response}
Correct Answer: {answer}
Question Complexity Level: {complexity}
Original Q/A Reference: {original}
Evaluation Aspects: {question_class}
""".strip()


def build_prompt(*args, **kwargs):
    messages = [
        {"role": "user", "content": build_adjudicator_prompt(*args, **kwargs)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_json_safe(text):
    _json_re = re.compile(r"\{.*\}", re.DOTALL)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _json_re.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        tqdm.write(str(text))
        return None


@torch.inference_mode()
def judge_batch(prompts):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
    )
    gen_ids = gen_ids[::, inputs["input_ids"].shape[-1] : :]

    outs = []
    for i in range(len(gen_ids)):
        gen_slice = gen_ids[i]
        text = tokenizer.decode(gen_slice, skip_special_tokens=True).strip()
        outs.append(parse_json_safe(text))
    return outs


df = pd.read_csv("data/test.csv")

reader = ModelUtils.TestLogger.ResultsReader(file_path=config["test_output_file"])

labels = list(reader.labels)
preds = list(reader.predictions)

n_samples = len(labels)
batch_size = config["batch_size"]

pbar = tqdm(
    range(0, n_samples, batch_size), total=(n_samples + batch_size - 1) // batch_size
)
results = []


def get_acc():
    score = []
    for j in results:
        score.append(j["score"])
    return np.mean(score)


for start in pbar:
    end = min(start + batch_size, n_samples)
    batch = list(
        zip(
            df["question"][start:end:],
            preds[start:end:],
            df["complexity"][start:end:],
            df["original"][start:end:],
            df["question_class"][start:end:],
            labels[start:end:],
        )
    )

    batch_res = judge_batch([build_prompt(*x) for x in batch])

    for res in batch_res:
        results.append(res)

    pbar.set_postfix(
        accuracy = round(get_acc(), 4),
    )


results_df = pd.DataFrame(
    {
        "img_id": df["img_id"],
        "question": reader.questions,
        "label": reader.labels,
        "prediction": reader.predictions,
        "judge": results,
        "complexity": df["complexity"],
        "question_class": df["question_class"],
    }
)

results_df.to_csv(config["output_file"], index = False)