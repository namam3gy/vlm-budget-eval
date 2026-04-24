#!/usr/bin/env python3
"""Budget-constrained sequential information-seeking evaluation.

At each step the model observes:
  - the question and multiple-choice options,
  - the text sentences and image tiles revealed so far,
  - the remaining budget and how many more hints are available per modality,
and outputs ONE JSON action:
  {"action": "ANSWER", "choice": "A"}        -> commit to a choice; episode ends
  {"action": "REQUEST_TEXT"}                 -> reveal next text sentence (-1 budget)
  {"action": "REQUEST_VISUAL"}               -> reveal next image tile   (-1 budget)

If the model exhausts the budget without answering, a forced-answer turn is
issued (no extra budget, but no new info). The per-sample outputs record the
final choice, correctness, modality mix, and the full action trace.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    preproc_dir: Path = Path("preproc")
    out_dir: Path = Path("output")
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    torch_dtype: str = "bfloat16"
    budget: int = 6
    max_new_tokens: int = 128
    temperature: float = 0.0
    random_seed: int = 42
    max_eval_rows: int = -1
    tile_order: str = "shuffled"   # "shuffled" | "row_major"
    text_order: str = "natural"     # "natural" | "shuffled"
    save_trace: bool = True
    # If the model asks for an exhausted modality or emits unparseable JSON:
    #   consume 1 budget as a "wasted" action, up to this many times, then force answer.
    max_wasted_before_force: int = 2
    # Once force-answer mode is engaged, try this many model turns; if the model
    # still refuses to ANSWER, end the episode with final_action="REFUSED_ANSWER".
    max_forced_attempts: int = 2
    # Which acting policy to use:
    #   "model"         — the model chooses ANSWER / REQUEST_TEXT / REQUEST_VISUAL each turn
    #   "always_text"   — auto-request text every turn until budget or text exhausted, then force answer
    #   "always_visual" — auto-request visual every turn until budget or tiles exhausted, then force answer
    #   "full_info"     — reveal ALL text sentences and ALL tiles upfront, single forced-answer turn
    policy: str = "model"
    # Override the default SYSTEM_INSTRUCTION (e.g., to test prompt nudges).
    system_instruction: Optional[str] = None


# ---------------------------------------------------------------------------
# Dtype helper
# ---------------------------------------------------------------------------

def get_dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _coerce_list(v: Any) -> List[Any]:
    """Parquet may hand us numpy arrays or JSON-ish strings; normalize to list."""
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if hasattr(v, "tolist"):
        return list(v.tolist())
    if isinstance(v, str):
        try:
            return list(json.loads(v))
        except Exception:
            try:
                return list(ast.literal_eval(v))
            except Exception:
                return [v]
    return list(v)


def load_samples(preproc_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(preproc_dir / "samples.parquet")
    list_cols = ["choices", "choice_letters", "text_sentences", "tile_rel_paths", "tile_labels"]
    for c in list_cols:
        if c in df.columns:
            df[c] = df[c].map(_coerce_list)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model plumbing
# ---------------------------------------------------------------------------

def load_model_and_processor(cfg: EvalConfig):
    dtype = get_dtype(cfg.torch_dtype)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    return model, processor


SYSTEM_INSTRUCTION = (
    "You are answering a multiple-choice question under a limited information budget. "
    "At each turn you will see the question, the choices, whatever text sentences and "
    "image tiles have been revealed so far, and how much budget remains.\n"
    "\n"
    "You must output ONE JSON object on a single line and nothing else. The valid actions are:\n"
    '  {"action": "ANSWER", "choice": "A"}   (commit to a choice; ends the episode, costs 0)\n'
    '  {"action": "REQUEST_TEXT"}            (reveal the next text sentence; costs 1)\n'
    '  {"action": "REQUEST_VISUAL"}          (reveal the next image tile;   costs 1)\n'
    "\n"
    "Choose REQUEST_TEXT vs REQUEST_VISUAL based on which modality is more likely to "
    "contain the missing information for this question. Answer as soon as you are "
    "confident; wasted requests shrink your remaining budget."
)


def build_user_content(
    sample: Dict[str, Any],
    revealed_tiles: List[Tuple[int, str, Image.Image]],  # (tile_idx, label, PIL)
    revealed_text: List[str],
    remaining_budget: int,
    text_left: int,
    visual_left: int,
    force_answer: bool,
) -> List[Dict[str, Any]]:
    """Build the user message content (list of text/image blocks)."""
    content: List[Dict[str, Any]] = []

    header = [f"Question: {sample['question']}", "Choices:"]
    for letter, choice in zip(sample["choice_letters"], sample["choices"]):
        header.append(f"  {letter}) {choice}")
    content.append({"type": "text", "text": "\n".join(header)})

    if revealed_text:
        lines = [f"Text hints revealed ({len(revealed_text)}):"]
        for k, s in enumerate(revealed_text, 1):
            lines.append(f"  {k}. {s}")
        content.append({"type": "text", "text": "\n".join(lines)})
    else:
        content.append({"type": "text", "text": "Text hints revealed (0): (none yet)"})

    if revealed_tiles:
        content.append({
            "type": "text",
            "text": f"Image tiles revealed ({len(revealed_tiles)}), labeled by spatial position:",
        })
        for _t_idx, label, pil_img in revealed_tiles:
            content.append({"type": "text", "text": f"Tile [{label}]:"})
            content.append({"type": "image", "image": pil_img})
    else:
        content.append({"type": "text", "text": "Image tiles revealed (0): (none yet)"})

    status = [
        f"Remaining budget: {remaining_budget}",
        f"Text hints still available: {text_left}",
        f"Image tiles still available: {visual_left}",
    ]
    if force_answer:
        status.append(
            'You must answer NOW. Output {"action": "ANSWER", "choice": "<letter>"} only.'
        )
    else:
        status.append(
            "Output one JSON action and nothing else."
        )
    content.append({"type": "text", "text": "\n".join(status)})
    return content


@torch.inference_mode()
def generate_once(model, processor, messages, max_new_tokens: int, temperature: float) -> str:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0.0}
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature

    out_ids = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    new_tokens = out_ids[:, input_len:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip()


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_action(raw: str) -> Optional[Dict[str, Any]]:
    """Return a normalized action dict or None if unparseable."""
    if not raw:
        return None
    m = JSON_OBJ_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(m.group(0))
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None
    action = str(obj.get("action", "")).strip().upper()
    if action not in {"ANSWER", "REQUEST_TEXT", "REQUEST_VISUAL"}:
        return None
    out: Dict[str, Any] = {"action": action}
    if action == "ANSWER":
        choice = obj.get("choice")
        if choice is None:
            choice = obj.get("answer")
        if choice is None:
            return None
        out["choice"] = str(choice).strip().upper()[:1]
    return out


# ---------------------------------------------------------------------------
# Per-sample episode
# ---------------------------------------------------------------------------

def _run_full_info(model, processor, sample: Dict[str, Any], cfg: EvalConfig, rng: random.Random) -> Dict[str, Any]:
    """Ceiling baseline: reveal ALL text sentences + ALL image tiles upfront, ask once."""
    preproc_dir = Path(cfg.preproc_dir)
    n_tiles = int(sample["n_tiles"])
    n_sents = int(sample["n_sentences"])
    tile_labels = list(sample["tile_labels"])
    tile_paths = list(sample["tile_rel_paths"])
    sentences = list(sample["text_sentences"])

    tile_order = list(range(n_tiles))
    if cfg.tile_order == "shuffled":
        rng.shuffle(tile_order)
    text_order = list(range(n_sents))
    if cfg.text_order == "shuffled":
        rng.shuffle(text_order)

    revealed_tiles: List[Tuple[int, str, Image.Image]] = []
    for t_idx in tile_order:
        img = Image.open(preproc_dir / tile_paths[t_idx]).convert("RGB")
        revealed_tiles.append((t_idx, tile_labels[t_idx], img))
    revealed_text = [sentences[i] for i in text_order]

    content = build_user_content(
        sample=sample,
        revealed_tiles=revealed_tiles,
        revealed_text=revealed_text,
        remaining_budget=0,
        text_left=0,
        visual_left=0,
        force_answer=True,
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": cfg.system_instruction or SYSTEM_INSTRUCTION}]},
        {"role": "user", "content": content},
    ]
    raw = generate_once(model, processor, messages, cfg.max_new_tokens, cfg.temperature)
    parsed = parse_action(raw)

    final_action = "FORCED_ANSWER" if parsed and parsed["action"] == "ANSWER" else "PARSE_FAIL"
    final_choice = parsed.get("choice") if parsed and parsed["action"] == "ANSWER" else None
    correct_letter = str(sample["answer_letter"]).upper()
    is_correct = (final_choice is not None and final_choice.upper() == correct_letter)

    trace = [{
        "step": 1, "budget_before": 0,
        "text_left_before": 0, "visual_left_before": 0,
        "raw": raw, "forced": True,
        "action": final_action,
        "revealed_text_idxs": [int(i) for i in text_order],
        "revealed_tile_idxs": [int(i) for i in tile_order],
    }]

    return {
        "sample_id": sample["sample_id"],
        "question": sample["question"],
        "answer_letter": correct_letter,
        "answer_idx": int(sample["answer_idx"]),
        "subject": sample.get("subject", ""),
        "topic": sample.get("topic", ""),
        "n_sentences": n_sents,
        "n_tiles": n_tiles,
        "budget": 0,
        "final_action": final_action,
        "final_choice": final_choice,
        "final_raw": raw,
        "is_correct": bool(is_correct),
        "budget_used": 0,
        "text_requests": n_sents,
        "visual_requests": n_tiles,
        "wasted_requests": 0,
        "parse_failures": 0 if parsed else 1,
        "tile_reveal_order": [int(i) for i in tile_order],
        "text_reveal_order": [int(i) for i in text_order],
        "n_steps": 1,
        "trace": trace if cfg.save_trace else None,
    }


def run_episode(model, processor, sample: Dict[str, Any], cfg: EvalConfig, rng: random.Random) -> Dict[str, Any]:
    if cfg.policy == "full_info":
        return _run_full_info(model, processor, sample, cfg, rng)

    preproc_dir = Path(cfg.preproc_dir)

    n_tiles = int(sample["n_tiles"])
    n_sents = int(sample["n_sentences"])
    tile_labels = list(sample["tile_labels"])
    tile_paths = list(sample["tile_rel_paths"])
    sentences = list(sample["text_sentences"])

    tile_order = list(range(n_tiles))
    if cfg.tile_order == "shuffled":
        rng.shuffle(tile_order)

    text_order = list(range(n_sents))
    if cfg.text_order == "shuffled":
        rng.shuffle(text_order)

    revealed_tiles: List[Tuple[int, str, Image.Image]] = []
    revealed_text: List[str] = []
    trace: List[Dict[str, Any]] = []

    budget_left = int(cfg.budget)
    wasted = 0
    parse_failures = 0
    text_requests = 0
    visual_requests = 0
    forced_attempts = 0

    final_action: Optional[str] = None
    final_choice: Optional[str] = None
    final_raw: Optional[str] = None
    force_next = False

    # Hard backstop so a misbehaving model can't spin the loop forever.
    max_steps = max(int(cfg.budget), 1) + cfg.max_wasted_before_force + cfg.max_forced_attempts + 4

    step = 0
    while step < max_steps:
        step += 1
        text_left = n_sents - len(revealed_text)
        visual_left = n_tiles - len(revealed_tiles)

        is_forced = force_next or budget_left <= 0

        # Fixed-policy baselines pre-empt the model's action choice on non-forced turns.
        # On forced turns we still ask the model for its ANSWER.
        synth_action = None
        if not is_forced and cfg.policy == "always_text":
            if text_left > 0:
                synth_action = {"action": "REQUEST_TEXT"}
            else:
                is_forced = True  # nothing left to auto-reveal -> force answer
        elif not is_forced and cfg.policy == "always_visual":
            if visual_left > 0:
                synth_action = {"action": "REQUEST_VISUAL"}
            else:
                is_forced = True

        if synth_action is not None:
            raw = f"[policy={cfg.policy}] {json.dumps(synth_action)}"
            parsed = synth_action
        else:
            content = build_user_content(
                sample=sample,
                revealed_tiles=revealed_tiles,
                revealed_text=revealed_text,
                remaining_budget=budget_left,
                text_left=text_left,
                visual_left=visual_left,
                force_answer=is_forced,
            )
            messages = [
                {"role": "system", "content": [{"type": "text", "text": cfg.system_instruction or SYSTEM_INSTRUCTION}]},
                {"role": "user", "content": content},
            ]
            raw = generate_once(model, processor, messages, cfg.max_new_tokens, cfg.temperature)
            parsed = parse_action(raw)

        step_entry: Dict[str, Any] = {
            "step": step,
            "budget_before": budget_left,
            "text_left_before": text_left,
            "visual_left_before": visual_left,
            "raw": raw,
            "forced": is_forced,
        }

        if parsed is None:
            parse_failures += 1
            step_entry["action"] = "PARSE_FAIL"
            if is_forced:
                # give up; record empty final
                trace.append(step_entry)
                final_action = "PARSE_FAIL"
                final_choice = None
                final_raw = raw
                break
            wasted += 1
            budget_left -= 1
            trace.append(step_entry)
            if wasted >= cfg.max_wasted_before_force:
                force_next = True
            continue

        act = parsed["action"]
        step_entry["action"] = act

        if is_forced:
            # After force prompt, only ANSWER is honored.
            if act == "ANSWER":
                final_action = "FORCED_ANSWER"
                final_choice = parsed.get("choice")
                final_raw = raw
                trace.append(step_entry)
                break
            # Model refused to answer despite force prompt — give it a few tries.
            forced_attempts += 1
            step_entry["action"] = f"{act}_IGNORED_FORCED"
            trace.append(step_entry)
            if forced_attempts >= cfg.max_forced_attempts:
                final_action = "REFUSED_ANSWER"
                final_choice = None
                final_raw = raw
                break
            force_next = True
            continue

        if act == "ANSWER":
            final_action = "ANSWER"
            final_choice = parsed.get("choice")
            final_raw = raw
            trace.append(step_entry)
            break

        if act == "REQUEST_TEXT":
            if text_left <= 0:
                wasted += 1
                budget_left -= 1
                step_entry["action"] = "REQUEST_TEXT_WASTED"
                trace.append(step_entry)
            else:
                next_sent_idx = text_order[len(revealed_text)]
                revealed_text.append(sentences[next_sent_idx])
                text_requests += 1
                budget_left -= 1
                step_entry["revealed_text_idx"] = int(next_sent_idx)
                trace.append(step_entry)
        elif act == "REQUEST_VISUAL":
            if visual_left <= 0:
                wasted += 1
                budget_left -= 1
                step_entry["action"] = "REQUEST_VISUAL_WASTED"
                trace.append(step_entry)
            else:
                next_tile_idx = tile_order[len(revealed_tiles)]
                label = tile_labels[next_tile_idx]
                img = Image.open(preproc_dir / tile_paths[next_tile_idx]).convert("RGB")
                revealed_tiles.append((next_tile_idx, label, img))
                visual_requests += 1
                budget_left -= 1
                step_entry["revealed_tile_idx"] = int(next_tile_idx)
                step_entry["revealed_tile_label"] = label
                trace.append(step_entry)

        if wasted >= cfg.max_wasted_before_force:
            force_next = True

        if budget_left <= 0 and final_action is None:
            force_next = True
            # loop continues; next iteration is forced

    if final_action is None:
        # Hit the hard step cap without ever resolving — treat as refused.
        final_action = "REFUSED_ANSWER"

    correct_letter = str(sample["answer_letter"]).upper()
    is_correct = (final_choice is not None and final_choice.upper() == correct_letter)

    return {
        "sample_id": sample["sample_id"],
        "question": sample["question"],
        "answer_letter": correct_letter,
        "answer_idx": int(sample["answer_idx"]),
        "subject": sample.get("subject", ""),
        "topic": sample.get("topic", ""),
        "n_sentences": n_sents,
        "n_tiles": n_tiles,
        "budget": int(cfg.budget),
        "final_action": final_action,
        "final_choice": final_choice,
        "final_raw": final_raw,
        "is_correct": bool(is_correct),
        "budget_used": int(cfg.budget) - budget_left,
        "text_requests": text_requests,
        "visual_requests": visual_requests,
        "wasted_requests": wasted,
        "parse_failures": parse_failures,
        "tile_reveal_order": [int(i) for i in tile_order[:len(revealed_tiles)]],
        "text_reveal_order": [int(i) for i in text_order[:len(revealed_text)]],
        "n_steps": len(trace),
        "trace": trace if cfg.save_trace else None,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(pred_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if pred_df.empty:
        return {"overall": pd.DataFrame(), "by_subject": pd.DataFrame()}

    overall = pd.DataFrame([{
        "n": len(pred_df),
        "accuracy": float(pred_df["is_correct"].mean()),
        "mean_budget_used": float(pred_df["budget_used"].mean()),
        "mean_text_requests": float(pred_df["text_requests"].mean()),
        "mean_visual_requests": float(pred_df["visual_requests"].mean()),
        "mean_wasted": float(pred_df["wasted_requests"].mean()),
        "parse_fail_rate": float((pred_df["parse_failures"] > 0).mean()),
        "forced_answer_rate": float((pred_df["final_action"] == "FORCED_ANSWER").mean()),
        "answered_without_info_rate": float(
            ((pred_df["text_requests"] == 0) & (pred_df["visual_requests"] == 0)).mean()
        ),
    }])

    by_subject = (
        pred_df.groupby(pred_df["subject"].fillna("unknown"), dropna=False)
        .agg(
            n=("sample_id", "count"),
            accuracy=("is_correct", "mean"),
            mean_budget_used=("budget_used", "mean"),
            mean_text_requests=("text_requests", "mean"),
            mean_visual_requests=("visual_requests", "mean"),
        )
        .reset_index()
        .rename(columns={"subject": "subject"})
    )

    return {"overall": overall, "by_subject": by_subject}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Optional[EvalConfig] = None):
    if cfg is None:
        cfg = parse_args()

    preproc_dir = Path(cfg.preproc_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_df = load_samples(preproc_dir)
    if cfg.max_eval_rows > 0:
        samples_df = samples_df.head(cfg.max_eval_rows).reset_index(drop=True)
    print(f"Eval samples: {len(samples_df)}")

    print(f"Loading model {cfg.model_id} ...")
    model, processor = load_model_and_processor(cfg)

    rng = random.Random(cfg.random_seed)
    results: List[Dict[str, Any]] = []
    for row in tqdm(samples_df.to_dict(orient="records"), total=len(samples_df)):
        results.append(run_episode(model, processor, row, cfg, rng))

    pred_df = pd.DataFrame(results)

    # Drop trace from parquet/csv (nested dicts are awkward) but keep in jsonl.
    flat_df = pred_df.drop(columns=[c for c in ["trace"] if c in pred_df.columns])
    flat_df.to_parquet(out_dir / "predictions.parquet", index=False)
    flat_df.to_csv(out_dir / "predictions.csv", index=False)
    pred_df.to_json(out_dir / "predictions.jsonl", orient="records", lines=True, force_ascii=False)

    aggs = aggregate(flat_df)
    for name, adf in aggs.items():
        adf.to_csv(out_dir / f"summary_{name}.csv", index=False)
        adf.to_json(out_dir / f"summary_{name}.jsonl", orient="records", lines=True, force_ascii=False)

    print("\n=== Overall ===")
    print(aggs["overall"].to_string(index=False) if not aggs["overall"].empty else "no rows")
    print("\n=== By subject ===")
    print(aggs["by_subject"].to_string(index=False) if not aggs["by_subject"].empty else "no rows")


def parse_args(argv=None) -> EvalConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--preproc-dir", type=Path, default=Path("preproc"))
    p.add_argument("--out-dir", type=Path, default=Path("output"))
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--budget", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--max-eval-rows", type=int, default=-1)
    p.add_argument("--tile-order", default="shuffled", choices=["shuffled", "row_major"])
    p.add_argument("--text-order", default="natural", choices=["natural", "shuffled"])
    p.add_argument("--no-trace", action="store_true")
    args = p.parse_args(argv)
    kwargs = {k.replace("-", "_"): v for k, v in vars(args).items() if k != "no_trace"}
    kwargs["save_trace"] = not args.no_trace
    return EvalConfig(**kwargs)


if __name__ == "__main__":
    main()
