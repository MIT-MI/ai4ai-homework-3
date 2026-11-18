"""One training script for SFT and self-distillation. TODO 4 and 5 are yours.

    python train.py --run runs/sft  --data demo    --loss ce                      # SFT
    python train.py --run runs/sdft --data student --loss fkl                     # SDFT (on-policy self-distillation)
    python train.py --run runs/sdft_mine --data student --loss fkl --context mine # SDFT with your own c

A method is two choices:

    --data   who writes the response y that we train on?
             demo      the expert: y is the demonstration                              off-policy
             student   the student: y ~ pi_theta( . | x)                               on-policy
    --loss   what does the student learn at each token of y?
             ce        to predict y itself                                             hard target
             fkl       the teacher's distribution, KL(p_T || p_S)                      soft target
             rkl       the teacher's distribution, KL(p_S || p_T)                      soft target

The teacher is the base model reading the privileged information c (--context, see contexts.py). It is not trained,
so it is exactly the teacher that analyze.py measures.

Each step:   1. sample_response()   who writes y?                       TODO 4
             2. compute_loss()      what does the student learn on y?   TODO 5
             3. update the student

Debug on a CPU in two minutes with a tiny model:

    python train.py --run runs/debug --data student --loss fkl --model HuggingFaceTB/SmolLM2-135M-Instruct \
        --n-train 16 --n-eval 8 --batch-size 4 --max-new-tokens 32
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import get_cosine_schedule_with_warmup

import losses
from contexts import CONTEXTS
from hw3 import data
from hw3.evaluate import evaluate
from hw3.model import base_weights, chat, encode, generate, load, lora, masked_mean, response_logits, weights


def sample_response(args, model, tok, batch: list[dict]) -> list[list[int]]:
    """1. Who writes the response y that the student learns from?

    For SFT (--data demo) it is the expert's demonstration, which is done for you.
    TODO 4: for SDFT (--data student), let the student write it. Sample one response for each example from the
    current model (it holds the student's weights), with

        generate(model, tok, prompts, max_new_tokens, sample=True)  ->  one list of token ids per prompt

    Args:
        args: the command-line arguments; you need args.data and args.max_new_tokens.
        model, tok: the model and its tokenizer.
        batch: a list of examples. batch[i]["x"] is the student's prompt, already formatted as text.
    Returns:
        A list with one response per example, each a list of token ids."""
    if args.data == "demo":
        return [(encode(tok, ex["demo"]) + [tok.eos_token_id])[: args.max_new_tokens] for ex in batch]
    raise NotImplementedError("TODO 4")


def compute_loss(args, model, tok, base, batch: list[dict], responses: list[list[int]]) -> torch.Tensor:
    """2. What does the student learn from y? TODO 5.

    1. Run the student on its prompt and the response, to get its logits at every token of the response:

           student_logits, y, mask = response_logits(model, tok, prompts, responses)

       student_logits is [B, T, V], y [B, T] the response's tokens, and mask [B, T] is False where a response is
       padded. The prompts are [ex["x"] for ex in batch].
    2. For SFT (args.loss == "ce") the loss is the cross-entropy of y. Otherwise, run the teacher the same way: it
       reads [ex["xc"] for ex in batch] (the prompt with c) and has the base model's weights, which you get with
       `with weights(model, base):`. The teacher is not trained, so also wrap it in `torch.no_grad()`.
    3. Take the per-token loss from losses.py ("fkl": forward_kl, "rkl": reverse_kl) and average it over the real
       tokens with masked_mean(per_token, mask).

    Args:
        args: the command-line arguments; you need args.loss ("ce", "fkl" or "rkl").
        model, tok: the model, which holds the student's weights, and its tokenizer.
        base: the base model's weights, for the teacher.
        batch: a list of examples, each with "x" (the student's prompt) and "xc" (the teacher's prompt).
        responses: one response per example, as token ids (from sample_response).
    Returns:
        The loss, a scalar tensor with gradients."""
    raise NotImplementedError("TODO 5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="output folder, e.g. runs/sdft")
    ap.add_argument("--data", choices=["demo", "student"], required=True)
    ap.add_argument("--loss", choices=["ce", "fkl", "rkl"], required=True)
    ap.add_argument("--context", choices=list(CONTEXTS), default="demo")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-train", type=int, default=2000, help="training examples (one pass)")
    ap.add_argument("--n-eval", type=int, default=None, help="evaluation examples (default: all)")
    ap.add_argument("--batch-size", type=int, default=16, help="examples per weight update")
    ap.add_argument("--micro-batch", type=int, default=4, help="examples per forward pass; lower it if out of memory")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-new-tokens", type=int, default=256, help="length limit of a training response")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    (args.run / "args.json").write_text(json.dumps(vars(args), default=str, indent=1))
    torch.manual_seed(args.seed)

    model, tok = load(args.model)
    base = base_weights(model)                               # the teacher's weights: the base model's
    examples = data.load("train", args.n_train)
    for ex in examples:
        ex["x"] = chat(tok, ex["prompt"])                    # what the student reads

    steps = len(examples) // args.batch_size
    optimizer = torch.optim.AdamW(lora(model).values(), lr=args.lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=min(10, steps // 10), num_training_steps=steps)
    log, start = open(args.run / "log.jsonl", "w"), time.time()

    for step in range(steps):
        batch = examples[step * args.batch_size : (step + 1) * args.batch_size]
        responses = sample_response(args, model, tok, batch)                             # 1. who writes y?
        for ex, r in zip(batch, responses):                                              #    what the teacher reads,
            ex["xc"] = chat(tok, CONTEXTS[args.context](ex, tok.decode(r, skip_special_tokens=True)))  # given y
        loss = 0.0
        for i in range(0, len(batch), args.micro_batch):                                 # 2. what is learned on y?
            micro = slice(i, i + args.micro_batch)
            part = compute_loss(args, model, tok, base, batch[micro], responses[micro]) * len(batch[micro]) / len(batch)
            part.backward()
            loss += part.item()
        torch.nn.utils.clip_grad_norm_(lora(model).values(), 1.0)                        # 3. update the student
        optimizer.step(), scheduler.step(), optimizer.zero_grad()

        row = {"step": step + 1, "loss": loss, "length": sum(map(len, responses)) / len(responses)}
        log.write(json.dumps(row) + "\n"), log.flush()
        print(f"step {step + 1}/{steps}  loss {loss:.3f}  length {row['length']:.0f}  [{(time.time() - start) / 60:.0f} min]", flush=True)

    model.save_pretrained(args.run / "adapter")
    evaluate(model, tok, args.run, args.n_eval)  # new-task accuracy and the distance to the base model


if __name__ == "__main__":
    main()
