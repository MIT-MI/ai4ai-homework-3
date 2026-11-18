"""The tool-use data (ToolAlpaca, as prepared by the SDFT authors).

    python -m hw3.data        # downloads it and writes data/{train,heldout,test}.jsonl

Each line is one example:

    id              e.g. "train-17"
    api             the API's name; no API is shared between the three files
    prompt          x: the tool documentation, the answer format and the user's question
    answer          the correct tool calls: [{"Action": ..., "Action_Input": "<json>"}, ...]
    demo            the expert's response (Thought / Action / Action Input)        not in test.jsonl
    other_question  another question about the same API, and the expert's
    other_demo      response to it                                                 not in test.jsonl

    train.jsonl     what you train on
    heldout.jsonl   APIs that we took out of the training set; has demonstrations, so analyze.py uses it
    test.jsonl      the authors' 97 test examples; no demonstrations

New-task accuracy is measured on heldout + test.
"""

import json
import random
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
SOURCE = "https://raw.githubusercontent.com/idanshen/Self-Distillation/d77573212fa0a3ae2eeb64b9b44db1c251f75e3e/data/tooluse_data"
N_HELDOUT = 300        # at least this many examples are held out, by whole APIs
MAX_DEMO_CHARS = 1200  # drops the 3% longest demonstrations (up to 14 tool calls), to bound memory


def load(split: str, n: int | None = None) -> list[dict]:
    """split is "train", "heldout", "test" or "eval" (= heldout + test)."""
    if not (DATA / "train.jsonl").exists():
        prepare()
    names = ["heldout", "test"] if split == "eval" else [split]
    rows = [json.loads(line) for name in names for line in open(DATA / f"{name}.jsonl")]
    return rows[:n] if n else rows


def prepare():
    from datasets import load_from_disk

    for part in ("train_data", "eval_data"):
        (DATA / "raw" / part).mkdir(parents=True, exist_ok=True)
        for file in ("data-00000-of-00001.arrow", "dataset_info.json", "state.json"):
            if not (path := DATA / "raw" / part / file).exists():
                print("downloading", part, file)
                urllib.request.urlretrieve(f"{SOURCE}/{part}/{file}", path)

    seen, by_api = set(), {}
    for r in load_from_disk(str(DATA / "raw" / "train_data")):
        demo = "\n".join(r["golden_response"]).strip()
        if r["prompt"] in seen or len(demo) > MAX_DEMO_CHARS:
            continue
        seen.add(r["prompt"])
        by_api.setdefault(r["name"], []).append(
            {"api": r["name"], "prompt": r["prompt"], "question": r["instruction"], "answer": r["golden_answer"], "demo": demo}
        )

    rng = random.Random(0)
    apis = sorted(api for api, rows in by_api.items() if len(rows) >= 2)  # every example needs an "other" one
    rng.shuffle(apis)
    train, heldout = [], []
    for api in apis:
        rows = by_api[api]
        for i, r in enumerate(rows):
            other = rows[(i + 1) % len(rows)]
            r["other_question"], r["other_demo"] = other["question"], other["demo"]
        (heldout if len(heldout) < N_HELDOUT else train).extend(rows)
    rng.shuffle(train)

    test = [
        {"api": r["name"], "prompt": r["prompt"], "answer": r["golden_answer"]}
        for r in load_from_disk(str(DATA / "raw" / "eval_data"))
    ]
    for name, rows in (("train", train), ("heldout", heldout), ("test", test)):
        with open(DATA / f"{name}.jsonl", "w") as f:
            for i, r in enumerate(rows):
                r = {"id": f"{name}-{i}"} | {k: v for k, v in r.items() if k != "question"}
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)} examples, {len({r['api'] for r in rows})} APIs")


if __name__ == "__main__":
    prepare()
