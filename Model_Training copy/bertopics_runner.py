#!/usr/bin/env python3
"""
bertopics_runner.py
--------------------
Variant of strategy_runner.py: classifies entities using a BERTopic-style
topic model first (24 topics), then does a single fixed-subclass LLM step
within that topic before falling back to the regular SPARQL-driven LLM loop.

Per entity:
  1. Run the topics BERT model (topics_model) to predict one of 24 topics
     (output label format: "topic_0" ... "topic_23").
  2. Look up the fixed candidate-subclass list for that topic (from the
     bertopic subclass breakdown CSV) and ask the LLM to pick one, in a
     single step -- no SPARQL fetch involved.
  3. Check divergence against the entity's true type(s).
  4. Continue with the regular LLM_2.loop() SPARQL-driven cascade, checking
     divergence at every subsequent step, same as before.
"""

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from datetime import datetime

from bert_cascade import CascadeNode, label_to_iri, run_bert_cascade
from LLM_2 import (
    DEFAULT_MODELS,
    _iri_to_label,
    build_prompt,
    call_llm,
    clean_label,
    is_entity_under_class,
    loop,
)


def parse_args():
    p = argparse.ArgumentParser(description="BERTopic + fixed-subclass LLM step + LLM cascade.")
    p.add_argument("--input", required=True, help="Path to input JSON file.")
    p.add_argument("--output", default="results.json", help="Path to write results JSON.")
    p.add_argument("--topics-model-dir", required=True,
                   help="Path to the trained topics_model directory (24-way classifier).")
    p.add_argument("--topics-csv", required=True,
                   help="Path to the bertopic subclass breakdown CSV "
                        "(columns: bertopic_topic, source_class, count).")
    p.add_argument("--backend", default="mistral", help="LLM backend for the fallback stage.")
    p.add_argument("--model", default=None, help="Model name (backend-specific).")
    p.add_argument("--endpoint", default="http://localhost:9004", help="SPARQL endpoint.")
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-base", default="https://api.openai.com/v1")
    p.add_argument("--subclass-limit", type=int, default=1000)
    p.add_argument("--entity-sample", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--max-depth", type=int, default=10,
                   help="Max LLM traversal depth after the fixed-subclass step.")
    p.add_argument("--device", default=None, help="Force a torch device (cpu/cuda/mps). Default: auto-detect.")
    p.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between entities.")
    p.add_argument("--limit", type=int, default=None, help="Max number of entities to process from the input file.")
    p.add_argument("--dry-run", action="store_true", help="Skip the LLM calls (BERT stages still run).")
    return p.parse_args()


def load_topic_subclasses(csv_path):
    """
    Reads the bertopic subclass breakdown CSV and groups source_class labels
    by topic number.

    Returns: dict[int, list[str]]  e.g. {0: ["Professional_Wrestler_Q13474373", ...], ...}
    """
    topic_map = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic = int(row["bertopic_topic"])
            source_class = row["source_class"].strip()
            topic_map[topic].append(source_class)
    return dict(topic_map)


def topic_label_to_index(label):
    """Converts a topics-model output label like 'bertopic_0' to the int 0."""
    return int(label.strip().lower().replace("bertopic_", ""))


def build_topic_subclass_options(topic_labels):
    """
    Converts a list of raw source_class strings (e.g. 'Professional_Wrestler_Q13474373')
    into the {"iri": ..., "label": ...} dict format build_prompt() expects.
    """
    return [
        {"iri": label_to_iri(name), "label": name}
        for name in topic_labels
    ]


class LLMArgs:
    """Mutable args object passed into LLM_2.loop() for each LLM step."""
    def __init__(self, base_args, iri, description):
        self.iri = iri
        self.entity = None
        self.description = description
        self.endpoint = base_args.endpoint
        self.backend = base_args.backend
        self.model = base_args.model
        self.api_base = base_args.api_base
        self.api_key = base_args.api_key
        self.subclass_limit = base_args.subclass_limit
        self.entity_sample = base_args.entity_sample
        self.temperature = base_args.temperature
        self.max_tokens = base_args.max_tokens
        self.dry_run = base_args.dry_run


def run_fixed_subclass_step(base_args, model, current_label, subclasses, description):
    """
    Single LLM classification step using a FIXED candidate-subclass list
    (no SPARQL fetch) -- used right after the topics model fires.

    Mirrors the response-parsing logic in LLM_2.loop(), but against a list
    we already have in hand rather than one fetched live from the endpoint.

    Returns the chosen subclass IRI, or "FINISHED"/"something went wrong rahhhh".
    """
    if not subclasses:
        print("No candidate subclasses for this topic -- cannot build a meaningful prompt.")
        return "FINISHED"

    prompt = build_prompt(
        current_class=current_label,
        subclasses=subclasses,
        entity_examples=[],
        entity_to_classify=None,
        description=description,
    )

    print("─── PROMPT (fixed topic subclasses) ───────────────────────────")
    print(prompt)
    print("─────────────────────────────────────────────────────────────\n")

    if base_args.dry_run:
        print("[dry-run] Skipping LLM call.")
        return "something went wrong rahhhh"

    print("⏳ Calling LLM (fixed topic subclasses) …")
    response = call_llm(
        backend=base_args.backend,
        prompt=prompt,
        model=model,
        api_base=base_args.api_base,
        api_key=base_args.api_key,
        temperature=base_args.temperature,
        max_tokens=base_args.max_tokens,
    )

    print("\n─── LLM RESPONSE ────────────────────────────────────────────")
    print(response)
    print("─────────────────────────────────────────────────────────────\n")

    response_lower = response.strip().lower()
    if response_lower == "finished":
        print("LLM indicated finished.")
        return "FINISHED"

    valid = {clean_label(sc["label"]).lower(): sc for sc in subclasses}

    m = re.search(r"\b(\d+)\b", response)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(subclasses):
            chosen = subclasses[idx]
            print(f"✅ Chosen subclass : {clean_label(chosen['label'])}")
            print(f"   IRI            : {chosen['iri']}")
            return chosen["iri"]
        else:
            print(f"⚠️  Index {idx+1} out of range ({len(subclasses)} subclasses).")

    for clean, sc in valid.items():
        if clean in response_lower:
            print(f"✅ Matched subclass : {clean_label(sc['label'])}")
            print(f"   IRI             : {sc['iri']}")
            return sc["iri"]

    print(f"⚠️  Response '{response.strip()}' did not match any valid subclass.")
    return "something went wrong rahhhh"


def run_llm_stage(base_args, model, start_iri, entity_iri, description, max_depth):
    """
    Regular SPARQL-driven LLM loop: keep asking the LLM to pick the next
    most-specific subclass, starting at `start_iri`, until it says
    FINISHED, errors out, diverges, or hits max_depth.
    """
    args = LLMArgs(base_args, start_iri, description)
    current_label = _iri_to_label(start_iri)
    path = []
    diverged = False
    final_iri = start_iri

    for depth in range(max_depth):
        result = loop(args, model, current_label)

        if result in ("something went wrong rahhhh", "FINISHED"):
            break

        final_iri = result
        step_label = _iri_to_label(result)

        try:
            on_path = is_entity_under_class(base_args.endpoint, entity_iri, result)
        except Exception as e:
            print(f"    ⚠️  Divergence check failed ({e}); assuming on-path.")
            on_path = True

        path.append({
            "stage": "llm",
            "step": depth + 1,
            "iri": result,
            "label": step_label,
            "on_path": on_path,
        })

        if not on_path:
            diverged = True
            print(f"    ⚠️  Diverged at LLM step {depth + 1}: "
                  f"'{step_label}' is not an ancestor of the entity's true type(s).")
            break

        args.iri = result
        current_label = step_label

    return final_iri, path, diverged


def classify_entity(base_args, model, topics_tree, topic_subclasses, entity_iri, text, max_depth):
    """
    Full pipeline for one entity:
      1. topics BERT model -> topic number
      2. fixed-subclass LLM step within that topic
      3. divergence check
      4. regular SPARQL-driven LLM loop, checking divergence at every step

    Returns (final_iri, path, diverged, topic_label, elapsed).
    """
    start = time.perf_counter()
    path = []
    diverged = False

    # --- Step 1: topics BERT model -------------------------------------
    topic_steps, topic_label = run_bert_cascade(topics_tree, text, device=base_args.device)
    topic_step_info = topic_steps[-1] if topic_steps else {"confidence": None, "model_dir": None}

    path.append({
        "stage": "bert_topic",
        "step": 1,
        "topic_label": topic_label,
        "confidence": (round(topic_step_info["confidence"], 4)
                       if topic_step_info["confidence"] is not None else None),
        "model_dir": topic_step_info["model_dir"],
    })

    topic_idx = topic_label_to_index(topic_label) #TODO FIX THIS (it has already been fixed for the testing code. this was not run locally)
    print(f"    → topics model predicted: {topic_label} (index {topic_idx})")
    raw_subclasses = topic_subclasses.get(topic_idx, [])
    fixed_subclasses = build_topic_subclass_options(raw_subclasses)

    # --- Step 2: fixed-subclass LLM step within the topic ---------------
    fixed_result = run_fixed_subclass_step(base_args, model, topic_label, fixed_subclasses, text)

    if fixed_result in ("something went wrong rahhhh", "FINISHED"):
        elapsed = time.perf_counter() - start
        return topic_label, path, diverged, topic_label, elapsed

    step_label = _iri_to_label(fixed_result)
    try:
        on_path = is_entity_under_class(base_args.endpoint, entity_iri, fixed_result)
    except Exception as e:
        print(f"    ⚠️  Divergence check failed ({e}); assuming on-path.")
        on_path = True

    path.append({
        "stage": "llm_topic_fixed",
        "step": 1,
        "iri": fixed_result,
        "label": step_label,
        "on_path": on_path,
    })

    if not on_path:
        diverged = True
        print(f"    ⚠️  Diverged at fixed-subclass step: "
              f"'{step_label}' is not an ancestor of the entity's true type(s).")
        elapsed = time.perf_counter() - start
        return fixed_result, path, diverged, topic_label, elapsed

    # --- Step 3: regular SPARQL-driven LLM loop --------------------------
    final_iri, llm_path, llm_diverged = run_llm_stage(
        base_args, model, fixed_result, entity_iri, text, max_depth
    )
    path.extend(llm_path)
    diverged = diverged or llm_diverged

    elapsed = time.perf_counter() - start
    return final_iri, path, diverged, topic_label, elapsed


def run():
    base_args = parse_args()
    model = base_args.model or DEFAULT_MODELS[base_args.backend]

    with open(base_args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entities = list(raw.items())
    if base_args.limit:
        entities = entities[: base_args.limit]

    print("Strategy: bertopics")
    print(f"Loaded {len(entities)} entities from {base_args.input}")
    print(f"Topics model: {base_args.topics_model_dir}")
    print(f"Topics CSV  : {base_args.topics_csv}")
    print(f"LLM fallback -> Backend: {base_args.backend} | Model: {model} | Delay: {base_args.delay}s\n")

    topic_subclasses = load_topic_subclasses(base_args.topics_csv)
    
    topics_tree = CascadeNode(model_dir=base_args.topics_model_dir, next={})

    results = []
    total_start = time.perf_counter()

    for i, (entity_iri, entry) in enumerate(entities):
        text = entry["description"]
        local_name = entry.get("local_name", entity_iri)
        true_depth = entry.get("actual_depth")
        true_parents = entry.get("direct_parents", [])

        print(f"[{i+1}/{len(entities)}] {local_name[:60]}  (true actual_depth={true_depth})")

        final_iri, path, diverged, topic_label, elapsed = entity_iri, [], False, None, 0.0
        for attempt in range(3):
            try:
                final_iri, path, diverged, topic_label, elapsed = classify_entity(
                    base_args, model, topics_tree, topic_subclasses, entity_iri, text, base_args.max_depth
                )
                break
            except Exception as e:
                last_error = str(e)
                if "429" in last_error or "rate" in last_error.lower():
                    wait = base_args.delay * (2 ** attempt) * 5  # 5s → 10s → 20s
                elif "503" in last_error or "unavailable" in last_error.lower():
                    wait = 30 * (attempt + 1)  # 30s → 60s → 90s
                else:
                    wait = 5 * (attempt + 1)  # 5s → 10s → 15s
                print(f"    ⚠️  Error (attempt {attempt+1}/3): {last_error[:80]}")
                print(f"        Retrying in {wait:.0f}s …")
                time.sleep(wait)
        else:
            print("    ❌ Failed after 3 retries, skipping.")

        result = {
            "index":               i,
            "entity_iri":          entity_iri,
            "local_name":          local_name,
            "true_actual_depth":   true_depth,
            "true_direct_parents": [p["iri"] for p in true_parents],
            "predicted_topic":     topic_label,
            "path":                path,
            "predicted_depth":     len(path),
            "diverged":            diverged,
            "final_class_iri":     final_iri,
            "final_class_label":   _iri_to_label(final_iri),
            "elapsed_s":           round(elapsed, 3),
        }
        results.append(result)
        print(f"    → final: {result['final_class_label']}  "
              f"(topic={topic_label}, diverged={diverged}, {elapsed:.2f}s)\n")

        if i < len(entities) - 1:
            time.sleep(base_args.delay)

    total_elapsed = time.perf_counter() - total_start
    avg = total_elapsed / len(entities) if entities else 0

    summary = {
        "timestamp":       datetime.now().isoformat(),
        "strategy":        "bertopics",
        "backend":         base_args.backend,
        "model":           model,
        "total_entities":  len(entities),
        "total_elapsed_s": round(total_elapsed, 3),
        "avg_elapsed_s":   round(avg, 3),
        "results":         results,
    }

    with open(base_args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Done. {len(entities)} entities in {total_elapsed:.1f}s (avg {avg:.2f}s each)")
    print(f"Results saved to {base_args.output}")


if __name__ == "__main__":
    run()