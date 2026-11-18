"""The model: loading it with LoRA, sampling from it, scoring a response with it, and the teacher's weights.

train.py uses five things from here:

    load(name)                            -> model, tokenizer
    generate(model, tok, prompts)         -> responses (token ids), sampled or greedy
    response_logits(model, tok, prompts, responses)
                                          -> logits [B, T, V], y [B, T], mask [B, T]
    base_weights(model)                   -> the base model's LoRA weights (zeros), for the teacher
    with weights(model, w): ...           run the model with the LoRA weights w instead of the student's

Only the LoRA weights are trained, so a set of weights is a small dict {parameter name: tensor}.
"""

import os
from contextlib import contextmanager

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

DEVICE = os.environ.get("HW3_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def load(name: str, adapter: str | None = None, rank: int = 16):
    """The model with a fresh LoRA adapter (which changes nothing until trained), or with a saved one."""
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"  # for generate(); response_logits() pads by hand
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(DEVICE)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        config = LoraConfig(r=rank, lora_alpha=2 * rank, target_modules="all-linear", task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # only in train mode
    model.eval()
    return model, tok


def chat(tok, user_message: str) -> str:
    """Wrap a user message in the model's chat format, ending where the assistant's response starts."""
    return tok.apply_chat_template([{"role": "user", "content": user_message}], tokenize=False, add_generation_prompt=True)


def encode(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False).input_ids


def end_ids(model, tok) -> set[int]:
    eos = model.generation_config.eos_token_id
    return {tok.eos_token_id} | set(eos if isinstance(eos, list) else [eos] if eos is not None else [])


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int, sample: bool, batch_size: int = 16) -> list[list[int]]:
    """One response per prompt, as token ids that end with the end-of-turn token (unless cut at max_new_tokens).

    sample=True draws from the model's own distribution (temperature 1, no top-k / top-p); sample=False is greedy."""
    stop = end_ids(model, tok)
    sampling = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0) if sample else dict(do_sample=False)
    config = GenerationConfig(
        max_new_tokens=max_new_tokens, repetition_penalty=1.0, eos_token_id=sorted(stop), pad_token_id=tok.pad_token_id, **sampling
    )
    responses = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
        out = model.generate(**enc, generation_config=config)
        for row in out[:, enc.input_ids.shape[1] :].tolist():
            end = next((t for t, token in enumerate(row) if token in stop), len(row) - 1)
            responses.append(row[: end + 1])
    return responses


def response_logits(model, tok, prompts: list[str], responses: list[list[int]]):
    """Run the model on prompt + response and return its next-token logits at the response's positions.

    logits[b, t] is the model's prediction for token t of response b, given the prompt and tokens < t.
    Returns logits [B, T, V] (float32), y [B, T] (the response tokens) and mask [B, T] (False where padded)."""
    prompt_ids = [encode(tok, p) for p in prompts]
    B, T = len(prompts), max(len(r) for r in responses)
    P = max(len(p) for p in prompt_ids)
    # Row b is [padding, prompt, response, padding]: every prompt ends at position P, so every response starts there.
    ids = torch.full((B, P + T), tok.pad_token_id, dtype=torch.long, device=DEVICE)
    attention = torch.zeros((B, P + T), dtype=torch.long, device=DEVICE)
    y = torch.zeros((B, T), dtype=torch.long, device=DEVICE)
    mask = torch.zeros((B, T), dtype=torch.bool, device=DEVICE)
    for b, (p, r) in enumerate(zip(prompt_ids, responses)):
        ids[b, P - len(p) : P + len(r)] = torch.tensor(p + r)
        attention[b, P - len(p) : P + len(r)] = 1
        y[b, : len(r)] = torch.tensor(r)
        mask[b, : len(r)] = True
    positions = (attention.cumsum(1) - 1).clamp(min=0)
    # With gradients, train mode turns on gradient checkpointing (there is no dropout, so nothing else changes).
    model.train(torch.is_grad_enabled())
    # The logits at position i predict token i + 1, so the response is predicted at positions P - 1 ... P + T - 2.
    # Only those are computed: the full [B, P + T, V] logits would not fit in memory.
    out = model(input_ids=ids, attention_mask=attention, position_ids=positions, logits_to_keep=T + 1, use_cache=False)
    model.eval()
    return out.logits[:, :-1].float(), y, mask


def masked_mean(per_token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The mean over the real (unpadded) tokens of a [B, T] tensor."""
    return (per_token * mask).sum() / mask.sum()


# --- weights: the student's are in the model; the teacher's and the base model's are swapped in when needed ---


def lora(model) -> dict[str, torch.nn.Parameter]:
    return {n: p for n, p in model.named_parameters() if "lora_" in n}


def base_weights(model) -> dict[str, torch.Tensor]:
    """LoRA weights of zero: the model before any training."""
    return {n: torch.zeros_like(p) for n, p in lora(model).items()}


@contextmanager
def weights(model, w: dict[str, torch.Tensor]):
    """Inside the block the model runs with the LoRA weights w; afterwards it has the student's again."""
    params = lora(model)
    student = {n: p.data for n, p in params.items()}
    for n, p in params.items():
        p.data = w[n]
    try:
        yield
    finally:
        for n, p in params.items():
            p.data = student[n]
