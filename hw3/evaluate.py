"""New-task evaluation: tool-use accuracy, and how far the model has moved from the base model.

    python -m hw3.evaluate runs/base        # the base model (no runs/base/adapter exists)
    python -m hw3.evaluate runs/sdft        # a trained run; train.py already does this at its end

Writes <run>/metrics.json and <run>/eval.jsonl (every response and whether it was correct).

    accuracy      greedy response, without any context c, on heldout + test (399 examples)
    kl_to_base    mean over tokens of KL( pi_theta( . | x, y_<t) || pi_0( . | x, y_<t) ), y = the model's own responses
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch

import losses
from hw3 import data
from hw3.model import base_weights, chat, generate, load, response_logits, weights

MAX_NEW_TOKENS = 512
N_KL = 100  # examples used for kl_to_base


def parse(response: str) -> tuple[Counter, dict]:
    """The tool calls in a response: the names after "Action:", and all "Action Input:" JSON objects merged."""
    actions = Counter(re.findall(r"Action:\s*(\w+)", response))
    inputs = {}
    for m in re.finditer(r"Action Input:\s*", response):
        try:
            value, _ = json.JSONDecoder().raw_decode(response, m.end())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            inputs.update(value)
    return actions, inputs


def correct(response: str, answer: list[dict]) -> bool:
    """True if the response calls exactly the right tools with exactly the right arguments (in any order)."""
    gold_inputs = {}
    for call in answer:
        try:
            gold_inputs.update(json.loads(call["Action_Input"]))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    actions, inputs = parse(response)
    return actions == Counter(call["Action"] for call in answer) and inputs == gold_inputs


@torch.no_grad()
def kl_to_base(model, tok, prompts: list[str], responses: list[list[int]], batch_size: int = 4) -> float:
    total, count = 0.0, 0
    for i in range(0, len(prompts), batch_size):
        p, r = prompts[i : i + batch_size], responses[i : i + batch_size]
        logits, _, mask = response_logits(model, tok, p, r)
        with weights(model, base_weights(model)):
            base_logits, _, _ = response_logits(model, tok, p, r)
        total += (losses.reverse_kl(logits, base_logits) * mask).sum().item()
        count += mask.sum().item()
    return total / count


def evaluate(model, tok, run: Path, n: int | None = None) -> dict:
    examples = data.load("eval", n)
    prompts = [chat(tok, ex["prompt"]) for ex in examples]
    responses = generate(model, tok, prompts, MAX_NEW_TOKENS, sample=False, batch_size=32)
    texts = [tok.decode(r, skip_special_tokens=True) for r in responses]
    right = [correct(t, ex["answer"]) for t, ex in zip(texts, examples)]
    metrics = {
        "accuracy": sum(right) / len(right),
        "n": len(right),
        "kl_to_base": kl_to_base(model, tok, prompts[:N_KL], responses[:N_KL]),
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=1))
    with open(run / "eval.jsonl", "w") as f:
        for ex, t, r in zip(examples, texts, right):
            f.write(json.dumps({"id": ex["id"], "correct": r, "response": t, "answer": ex["answer"]}) + "\n")
    print(json.dumps(metrics))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-eval", type=int, default=None)
    args = ap.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    adapter = args.run / "adapter"
    model, tok = load(args.model, adapter=str(adapter) if adapter.exists() else None)
    evaluate(model, tok, args.run, args.n_eval)
