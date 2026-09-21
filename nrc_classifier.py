#!/usr/bin/env python3
"""
NRC lexicon sentiment + emotion scorer — runs over existing LLM predictions.

For every review already classified by the LLM (predictions_<seed>_<n>.jsonl),
score each word against the NRC Emotion Lexicon (v0.92):

  * sentiment = compare summed positive vs negative word flags
  * primary emotion = the emotion (anger, anticipation, disgust, fear, joy,
    sadness, surprise, trust) with the highest summed flag score.

No model calls. Writes enriched records to predictions_nrc.jsonl and prints a
comparison of the NRC method vs the LLM classifier.

Usage:
    python3 nrc_classifier.py                 # uses predictions_7_50.jsonl
    python3 nrc_classifier.py --in X.jsonl
"""
import argparse
import json
import os
import re

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]

# NRC data rows start at this 1-based line (after the 7-line title + license header).
DATA_START = 29


def load_lexicon(path="NRC_emotion_lexicon.txt"):
    """Return (word->set(emotions), pos_words, neg_words) from the NRC file."""
    word_emo = {}
    pos = set()
    neg = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            if ln < DATA_START:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, cat, flag = parts
            if flag != "1":
                continue
            cat = cat.strip().lower()
            word = word.strip().lower()
            if cat == "positive":
                pos.add(word)
            elif cat == "negative":
                neg.add(word)
            elif cat in EMOTIONS:
                word_emo.setdefault(word, set()).add(cat)
    return word_emo, pos, neg


def tokens(title, text):
    """Lowercased alphanumeric word tokens from title + body."""
    blob = f"{title} {text}".lower()
    return re.findall(r"[a-z']+", blob)


def score_review(title, text, lexicon):
    word_emo, pos, neg = lexicon
    emo_totals = {e: 0 for e in EMOTIONS}
    pos_tot, neg_tot, hits, words = 0, 0, 0, 0
    for w in tokens(title, text):
        if not w:
            continue
        words += 1
        if w in pos:
            pos_tot += 1
        if w in neg:
            neg_tot += 1
        if w in word_emo:
            hits += 1
            for e in word_emo[w]:
                emo_totals[e] += 1

    if pos_tot > neg_tot:
        sentiment = "POSITIVE"
    elif neg_tot > pos_tot:
        sentiment = "NEGATIVE"
    elif pos_tot == neg_tot == 0:
        sentiment = "NEUTRAL"     # no lexicon words found
    else:
        sentiment = "NEUTRAL"     # pos == neg tie

    best = None
    for e in EMOTIONS:            # canonical order breaks ties deterministically
        if best is None or emo_totals[e] > emo_totals[best]:
            best = e
    primary = best if emo_totals[best] > 0 else None

    return {
        "emotion_totals": emo_totals,
        "nrc_sentiment": sentiment,
        "nrc_primary_emotion": primary,
        "lexicon_hits": hits,
        "total_words": words,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="predictions_42_50.jsonl",
                    help="LLM predictions file to enrich")
    ap.add_argument("--out", default="predictions_nrc.jsonl",
                    help="output enriched file")
    ap.add_argument("--lexicon", default="NRC_emotion_lexicon.txt")
    args = ap.parse_args()

    print("Loading NRC lexicon ...")
    lex = load_lexicon(args.lexicon)

    with open(args.inp) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if not rows:
        print("No rows to process."); return

    enriched = []
    for r in rows:
        nrc = score_review(r.get("title"), r.get("text"), lex)
        rec = dict(r)
        rec.update(nrc)
        enriched.append(rec)

    with open(args.out, "w") as f:
        for rec in enriched:
            f.write(json.dumps(rec) + "\n")

    # ---- Comparison: NRC sentiment vs LLM prediction ---------------------------
    from collections import Counter
    # All reviews: strict agreement of labels (LLM can also say NEUTRAL).
    strict = Counter((r["pred"], r["nrc_sentiment"]) for r in enriched)
    strict_agree = sum(1 for r in enriched if r["pred"] == r["nrc_sentiment"])
    # Decided subset: only reviews where NRC took a non-neutral side.
    decided = [r for r in enriched if r["nrc_sentiment"] in ("POSITIVE", "NEGATIVE")]
    agreed = sum(1 for r in decided if r["pred"] == r["nrc_sentiment"])

    emo_dist = Counter(r["nrc_primary_emotion"] for r in enriched)
    sent_dist = Counter(r["nrc_sentiment"] for r in enriched)
    pred_dist = Counter(r["pred"] for r in enriched)

    nd = len(decided)
    print(f"Processed {len(enriched)} reviews, wrote {args.out}")
    print(f"\nLLM prediction distribution  : {dict(pred_dist)}")
    print(f"NRC sentiment distribution   : {dict(sent_dist)}")
    print(f"Primary-emotion distribution : {dict(emo_dist)}")
    print(f"\nLLM-vs-NRC agreement (ALL)        : {strict_agree}/{len(enriched)} "
          f"({100*strict_agree/len(enriched):.1f}%)")
    print(f"LLM-vs-NRC agreement (NRC decided): {agreed}/{nd} "
          f"({100*agreed/nd if nd else 0:.1f}%)  (NRC neutral/tie excluded: "
          f"{len(enriched)-nd})")
    print("\nLLM x NRC sentiment confusion (decided only):")
    for key in sorted(set((r['pred'], r['nrc_sentiment']) for r in decided)):
        print(f"  LLM={key[0]:<8} NRC={key[1]:<8}  n={sum(1 for r in decided if (r['pred'],r['nrc_sentiment'])==key)}")

    # Ground-truth agreement for each method.
    llm_ok = sum(1 for r in enriched if r["pred"] == r["label"])
    nrc_ok = sum(1 for r in decided if r["nrc_sentiment"] == r["label"])
    print(f"\nAgreement with star-rating ground truth: LLM {llm_ok}/{len(enriched)} "
          f"({100*llm_ok/len(enriched):.1f}%) | NRC {nrc_ok}/{nd} "
          f"({100*nrc_ok/nd if nd else 0:.1f}%)")

    # LLM errors and what NRC says about them.
    print("\nLLM errors and what NRC says:")
    for r in enriched:
        if r["pred"] != r["label"]:
            print(f"  {r['title']!r} | true={r['label']} LLM={r['pred']} "
                  f"NRC={r['nrc_sentiment']} primary_emotion={r['nrc_primary_emotion']}")

    # LLM-vs-NRC disagreements (both non-neutral take).
    print("\nLLM-vs-NRC disagreements (NRC decided):")
    for r in decided:
        if r["pred"] != r["nrc_sentiment"]:
            print(f"  {r['title']!r} | true={r['label']} LLM={r['pred']} NRC={r['nrc_sentiment']} "
                  f"emotion={r['nrc_primary_emotion']}")
    return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
