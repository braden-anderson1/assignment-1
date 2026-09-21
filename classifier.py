"""
Gift card Amazon review classifier — LLM prompt-based (3-class).

Classifies each review as POSITIVE, NEUTRAL, or NEGATIVE using ONLY the `title`
and `text` columns. The rating is used strictly to build ground-truth labels for
evaluation; it is NEVER included in the prompt sent to the model.

Ground-truth labels (full split):
    rating 4-5 -> POSITIVE, rating 3 -> NEUTRAL, rating 1-2 -> NEGATIVE.

Sampling pulls a balanced (≈equal per class), stratified, random group from the
WHOLE file — fixed random seed so the same set comes up every run — rather than
reading the first N rows in order (which under-represents the rare NEUTRAL class).

Usage:
    python3 classifier.py                  # default 50 per class (150 total)
    python3 classifier.py --n 50 --seed 42 --fresh
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter

CLASSES = ("POSITIVE", "NEUTRAL", "NEGATIVE")

# ---- Model configuration -----------------------------------------------------
# (User-specified endpoint; keep these here so they're easy to change.)
BASE_URL = "http://dobolyi.com:9001/v1"
API_KEY  = "6418"
MODEL    = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
DATA_FILE = "Gift_Cards.jsonl"

SYSTEM_PROMPT = (
    "You are a sentiment classifier for Amazon gift card reviews. Classify each "
    "review as POSITIVE, NEUTRAL, or NEGATIVE using ONLY its title and its body "
    "text; ignore any other information. A review with mixed, factual, or "
    "balanced feelings is NEUTRAL. When the title and the body text appear to "
    "disagree, rely more heavily on the body text than on the title. Respond "
    "with exactly one word: POSITIVE, NEUTRAL, or NEGATIVE. Do not explain and "
    "do not add anything else."
)


def build_user_prompt(title: str, text: str) -> str:
    """Construct the review prompt. NOTE: rating and all other fields are excluded."""
    return (
        "Classify the sentiment of this Amazon gift card review.\n\n"
        f"Title: {title}\n"
        f"Text: {text}\n\n"
        "Sentiment (POSITIVE, NEUTRAL, or NEGATIVE):"
    )


def label_from_rating(rating: float) -> str | None:
    """Map a star rating to its class under the full 3-way split."""
    if rating is None:
        return None
    if rating >= 4:
        return "POSITIVE"
    if rating == 3:
        return "NEUTRAL"
    if rating <= 2:
        return "NEGATIVE"
    return None


def classify_review(title: str, text: str, max_retries: int = 5) -> str:
    """Call the LLM for one review; return a label from CLASSES."""
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(title, text)},
        ],
        "max_tokens": 1024,         # reasoning model needs a budget for its thinking
        "temperature": 0.0,
        "extra_body": {
            "skip_special_tokens": True,
        },
    }).encode()

    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode())
            content = payload["choices"][0]["message"].get("content") or ""
            label = extract_label(content)
            if label:
                return label
            last_err = f"no parseable label from: {content[:200]!r}"
        except Exception as e:  # network / HTTP / JSON errors
            last_err = str(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Classification failed after {max_retries} tries: {last_err}")


def extract_label(content: str) -> str | None:
    """Pull a clean POSITIVE/NEUTRAL/NEGATIVE token out of the model's answer."""
    m = re.search(r"\b(POSITIVE|NEUTRAL|NEGATIVE)\b", content, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def load_reviews(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_labeled_sample(n_per_class: int, seed: int) -> list[dict]:
    """Balanced, stratified sample pulled from the WHOLE file.

    Buckets every review into its class (POSITIVE/NEUTRAL/NEGATIVE via rating),
    shuffles each bucket with a fixed seed, takes up to `n_per_class` from each,
    then shuffles the combined set again. This avoids the first-N-in-order bias
    that would under-represent the rare NEUTRAL class.
    """
    buckets = {c: [] for c in CLASSES}
    for rec in load_reviews(DATA_FILE):
        rating = rec.get("rating")
        label = label_from_rating(rating)
        if label is None:
            continue
        title = (rec.get("title") or "").strip()
        text  = (rec.get("text") or "").strip()
        if not title and not text:
            continue
        buckets[label].append({"true": label, "title": title, "text": text})

    rng = random.Random(seed)
    for c in CLASSES:
        rng.shuffle(buckets[c])          # shuffle BEFORE sampling = no ordering bias
    sample = []
    for c in CLASSES:
        sample.extend(buckets[c][:min(n_per_class, len(buckets[c]))])
    rng.shuffle(sample)
    return sample


# Checkpoint file is namespaced per (seed, n) so a rerun of the SAME command
# resumes cleanly while a DIFFERENT batch never reuses another run's labels.
_LOADED = {}
_FRESH = False


def _load_checkpoint(out_file: str):
    if _FRESH:
        return
    if os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                r = json.loads(line)
                _LOADED[r["idx"]] = r["pred"]


def evaluate(sample: list[dict], out_file: str) -> dict:
    _load_checkpoint(out_file)
    preds = [None] * len(sample)
    seen = set(_LOADED)
    failed = []

    def done(i, pred):
        preds[i] = pred
        with open(out_file, "a") as f:
            f.write(json.dumps({"idx": i, "label": sample[i]["true"],
                                "pred": pred, "title": sample[i]["title"],
                                "text": sample[i]["text"]}) + os.linesep)
        print(f"[{i + 1}/{len(sample)}] true={sample[i]['true']:<8} "
              f"pred={str(pred):<8} | {sample[i]['title'][:40]}", flush=True)

    # Pass 1: use checkpoint where available, else classify.
    for i, rec in enumerate(sample):
        if i in seen:
            preds[i] = _LOADED[i]
            continue
        try:
            done(i, classify_review(rec["title"], rec["text"]))
        except Exception as e:
            failed.append(i)
            print(f"[{i + 1}/{len(sample)}] FAILED: {e}", flush=True)

    # Pass 2: retry anything that failed on the first attempt.
    for i in failed:
        try:
            done(i, classify_review(sample[i]["title"], sample[i]["text"],
                                    max_retries=3))
        except Exception:
            print(f"[{i + 1}/{len(sample)}] still unclassified; skipped", flush=True)

    # ---- Multiclass metrics ------------------------------------------------
    classified = [i for i, p in enumerate(preds) if p is not None]
    n = len(classified)
    if n == 0:
        return {"n": 0}

    cm = Counter()                       # (true, predicted)
    for i in classified:
        cm[(sample[i]["true"], preds[i])] += 1

    correct = sum(cm[(c, c)] for c in CLASSES)
    acc = correct / n

    per_class = {}
    macro_f1 = 0.0
    for c in CLASSES:
        tp_ = cm[(c, c)]
        col = sum(cm[(r, c)] for r in CLASSES)   # predicted c
        row = sum(cm[(c, p)] for p in CLASSES)   # actually c
        prec = tp_ / col if col else 0.0
        rec  = tp_ / row if row else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"count": row,
                        "precision": round(prec, 4),
                        "recall": round(rec, 4),
                        "f1": round(f1, 4)}
        macro_f1 += f1
    macro_f1 /= len(CLASSES)

    confusion = {t: {p: cm[(t, p)] for p in CLASSES} for t in CLASSES}

    return {
        "n": n,
        "unclassified": len(sample) - n,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion": confusion,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50, help="reviews per class (default 50)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore the saved checkpoint and re-classify every review")
    args = ap.parse_args()

    out_file = f"predictions_{args.seed}_{args.n}.jsonl"
    if args.fresh:
        _FRESH = True
        _LOADED.clear()
        if os.path.exists(out_file):
            os.remove(out_file)

    print(f"Loading reviews from {DATA_FILE} ...", file=sys.stderr)
    sample = load_labeled_sample(args.n, args.seed)
    counts = "  ".join(f"{c} {sum(r['true']==c for r in sample)}" for c in CLASSES)
    print(f"Evaluation sample: {len(sample)} reviews  [{counts}]", file=sys.stderr)

    print("Classifying ...\n")
    metrics = evaluate(sample, out_file)

    print("\n===== RESULTS =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
