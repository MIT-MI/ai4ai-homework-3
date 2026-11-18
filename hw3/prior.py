"""Prior-task evaluation: does the model still do what it could do before training?

    python -m hw3.prior runs/base
    python -m hw3.prior runs/sdft

Runs two benchmarks that have nothing to do with tool use, with lm-evaluation-harness, and writes <run>/prior.json:

    ifeval      following formatting instructions ("answer in under 50 words", ...), 541 prompts; we report the
                share of prompts whose instructions were all followed (loose matching)
    humaneval   writing Python functions, 164 problems; pass@1 (the tests run the generated code on this machine)

These are the two prior tasks that SFT damaged most in the SDFT paper (Table 5). The LoRA adapter is merged into the
weights first: unmerged, generation is about 2.5 times slower.
"""

import argparse
import json
import os
from pathlib import Path

import torch

METRICS = {"ifeval": "prompt_level_loose_acc", "humaneval_instruct": "pass@1"}  # lm-eval appends ",<filter>" to these


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tasks", default="ifeval,humaneval_instruct")
    ap.add_argument("--limit", type=int, default=None, help="examples per task, for debugging")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    from hw3.model import load

    adapter = args.run / "adapter"
    model, tok = load(args.model, adapter=str(adapter) if adapter.exists() else None)
    model = model.merge_and_unload()  # a plain transformers model again, with the LoRA weights added in
    model.config.use_cache = True
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size, dtype=torch.bfloat16)
    out = lm_eval.simple_evaluate(
        model=lm, tasks=args.tasks.split(","), limit=args.limit, apply_chat_template=True, confirm_run_unsafe_code=True
    )

    prior = {}
    for task in args.tasks.split(","):
        key = next(k for k in out["results"][task] if k.startswith(METRICS[task] + ","))
        prior[task.removesuffix("_instruct")] = out["results"][task][key]
    prior["average"] = sum(prior.values()) / len(prior)
    (args.run / "prior.json").write_text(json.dumps(prior, indent=1))
    (args.run / "lm_eval.json").write_text(json.dumps(out["results"], indent=1, default=str))
    print(json.dumps(prior))


if __name__ == "__main__":
    main()
