"""Analyze kinds of privileged information c before any training (Part 1 and 3). Inference only.

    python analyze.py                                              # demo, answer, other and none
    python analyze.py --contexts demo mine --out runs/analyze_mine

The teacher is the base model reading x and c; the student is the base model reading x. On 100 held-out examples:

    summary.md                the teacher's accuracy, and the mean per-token KL(p_T || p_S) and KL(p_S || p_T)
                              on responses that the student sampled (the training losses at step 0)
    heatmap_<context>.html    those responses, each token colored by A_t = log p_T(y_t) - log p_S(y_t), with what the
                              teacher would have written where it disagrees most, and the teacher's own response
    teacher_<context>.jsonl   the teacher's own responses, and whether each is correct

Needs TODO 1 to 3.
"""

import argparse
import html
import json
from pathlib import Path

import torch

import losses
from contexts import CONTEXTS
from hw3 import data
from hw3.evaluate import MAX_NEW_TOKENS, correct
from hw3.model import chat, generate, load, response_logits


def page(tok, ex, tokens: list[int], s_logits, t_logits, adv, teacher_response: str) -> str:
    """One example as HTML: the student's response colored by A_t, where the teacher disagrees most and what it would
    write instead, and the teacher's own response. Hover over a token to see both probabilities."""
    p_s, p_t = s_logits.softmax(-1), t_logits.softmax(-1)
    word = lambda i: html.escape(repr(tok.decode([i]))[1:-1])  # a token, with spaces and newlines visible
    prefers = lambda t: ", ".join(f"{word(i)} {p:.2f}" for p, i in zip(*p_t[t].topk(3)))  # the teacher's top 3
    spans, rows = [], []
    for t, (token, a) in enumerate(zip(tokens, adv)):
        info = f"A = {a:+.2f}, student p = {p_s[t, token]:.3f}, teacher p = {p_t[t, token]:.3f}; teacher prefers {prefers(t)}"
        rgb = "33,102,172" if a > 0 else "178,24,43"
        text = html.escape(tok.decode([token])).replace("\n", "↵<br>")
        spans.append(f'<span style="background:rgba({rgb},{min(abs(a) / 6, 1):.2f})" title="{info}">{text}</span>')
    for t in sorted(range(len(tokens)), key=lambda t: adv[t])[:5]:  # the 5 tokens the teacher likes least
        rows.append(f"<tr><td>{word(tokens[t])}</td><td>{p_s[t, tokens[t]]:.3f}</td><td>{p_t[t, tokens[t]]:.3f}</td>"
                    f"<td>{adv[t]:+.2f}</td><td>{prefers(t)}</td></tr>")
    question = html.escape(ex["prompt"].split("Question:")[-1].strip())
    return (f"<h3>{ex['id']}: {question}</h3>"
            f'<p style="font-family:monospace;line-height:1.8">{"".join(spans)}</p>'
            "<table border=1 cellpadding=4 style='border-collapse:collapse'><tr><th>student wrote</th><th>student p</th>"
            "<th>teacher p</th><th>A</th><th>teacher would write (top 3)</th></tr>" + "".join(rows) + "</table>"
            f"<p>The teacher's own response:</p><pre>{html.escape(teacher_response)}</pre>")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", nargs="+", default=["demo", "answer", "other", "none"], choices=list(CONTEXTS))
    ap.add_argument("--out", type=Path, default=Path("runs/analyze"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    model, tok = load(args.model)
    examples = data.load("heldout", args.n)
    decode = lambda ids: tok.decode(ids, skip_special_tokens=True)
    x = [chat(tok, ex["prompt"]) for ex in examples]
    student = generate(model, tok, x, MAX_NEW_TOKENS, sample=False)
    student_acc = sum(correct(decode(r), ex["answer"]) for r, ex in zip(student, examples)) / len(examples)
    y = generate(model, tok, x, 256, sample=True)  # the responses that the student would train on

    rows = [f"Student (base model without c): accuracy {student_acc:.2f}\n", "| context | teacher accuracy | fkl | rkl |", "|---|---|---|---|"]
    for name in args.contexts:
        xc = [chat(tok, CONTEXTS[name](ex, decode(r))) for ex, r in zip(examples, y)]
        teacher = [decode(r) for r in generate(model, tok, xc, MAX_NEW_TOKENS, sample=False)]
        right = [correct(t, ex["answer"]) for t, ex in zip(teacher, examples)]
        with open(args.out / f"teacher_{name}.jsonl", "w") as f:
            for ex, t, r in zip(examples, teacher, right):
                f.write(json.dumps({"id": ex["id"], "correct": r, "response": t, "answer": ex["answer"]}) + "\n")

        fkl = rkl = n_tokens = 0.0
        pages = []
        for i in range(0, len(examples), 4):
            s_logits, y_ids, mask = response_logits(model, tok, x[i : i + 4], y[i : i + 4])
            t_logits, _, _ = response_logits(model, tok, xc[i : i + 4], y[i : i + 4])
            fkl += (losses.forward_kl(s_logits, t_logits) * mask).sum().item()
            rkl += (losses.reverse_kl(s_logits, t_logits) * mask).sum().item()
            n_tokens += mask.sum().item()
            if i < 8:  # heatmaps of the first 8 examples
                for j, (a, m) in enumerate(zip(losses.advantage(s_logits, t_logits, y_ids), mask)):
                    k = i + j
                    pages.append(page(tok, examples[k], y[k], s_logits[j][m], t_logits[j][m], a[m].tolist(), teacher[k]))
        rows.append(f"| {name} | {sum(right) / len(right):.2f} | {fkl / n_tokens:.3f} | {rkl / n_tokens:.3f} |")
        (args.out / f"heatmap_{name}.html").write_text(f'<meta charset="utf-8"><h2>context: {name}</h2>' + "".join(pages))
        print(rows[-1], flush=True)

    (args.out / "summary.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
