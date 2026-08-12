#!/usr/bin/env python3
"""
strategy_runner.py
-------------------
Shared driver for "BERT cascade, then hand off to an LLM" classification
strategies. Each strategy script (run_standard.py, run_layers.py, ...) builds
its own CascadeNode tree and calls `run(tree, strategy_name)`.

Per entity:
  1. Run the BERT cascade (see bert_cascade.py) starting at the strategy's
     root model, descending into sub-models as long as the predicted label
     has one.
  2. Once the cascade bottoms out at a label with no further BERT model,
     hand off to the LLM (LLM_2.loop), which keeps asking for the next most
     specific subclass via the SPARQL endpoint until it says FINISHED, fails,
     diverges from the entity's true type(s), or hits --max-depth.

At every step (BERT or LLM), the chosen class is checked against the
entity's real rdf:type(s) via SPARQL; the run stops early if it diverges.
"""

import argparse
import json
import time
from datetime import datetime

from bert_cascade import label_to_iri, run_bert_cascade
from LLM_2 import DEFAULT_MODELS, _iri_to_label, is_entity_under_class, loop


def parse_args():
    p = argparse.ArgumentParser(description="BERT-cascade + LLM entity classification.")
    p.add_argument("--input", required=True, help="Path to input JSON file.")
    p.add_argument("--output", default="results.json", help="Path to write results JSON.")
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
                   help="Max LLM traversal depth after the BERT cascade bottoms out.")
    p.add_argument("--device", default=None, help="Force a torch device (cpu/cuda/mps). Default: auto-detect.")
    p.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between entities.")
    p.add_argument("--limit", type=int, default=None, help="Max number of entities to process from the input file.")
    p.add_argument("--dry-run", action="store_true", help="Skip the LLM call (BERT stage still runs).")
    return p.parse_args()


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


def run_llm_stage(base_args, model, start_iri, entity_iri, description, max_depth):
    """
    Keep asking the LLM to pick the next most-specific subclass, starting at
    `start_iri`, until it says FINISHED, errors out, diverges from the
    entity's true type(s), or hits max_depth.
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


def classify_entity(base_args, model, tree, entity_iri, text, max_depth):
    """
    Run the full BERT cascade, then hand off to the LLM. Stops early if a
    BERT step already diverges from the entity's true type(s).

    Returns (final_iri, path, diverged, bert_depth, llm_depth, elapsed).
    """
    start = time.perf_counter()
    path = []
    diverged = False

    bert_steps, final_bert_label = run_bert_cascade(tree, text, device=base_args.device)

    for i, step in enumerate(bert_steps):
        iri = label_to_iri(step["label"])
        try:
            on_path = is_entity_under_class(base_args.endpoint, entity_iri, iri)
        except Exception as e:
            print(f"    ⚠️  Divergence check failed ({e}); assuming on-path.")
            on_path = True

        path.append({
            "stage": "bert",
            "step": i + 1,
            "iri": iri,
            "label": step["label"],
            "confidence": round(step["confidence"], 4),
            "model_dir": step["model_dir"],
            "on_path": on_path,
        })

        if not on_path:
            diverged = True
            print(f"    ⚠️  Diverged at BERT step {i + 1}: "
                  f"'{step['label']}' is not an ancestor of the entity's true type(s).")
            elapsed = time.perf_counter() - start
            return iri, path, diverged, len(path), 0, elapsed

    bert_depth = len(path)

    # BERT cascade bottomed out without diverging -- hand off to the LLM.
    start_iri = label_to_iri(final_bert_label)
    final_iri, llm_path, llm_diverged = run_llm_stage(
        base_args, model, start_iri, entity_iri, text, max_depth
    )
    path.extend(llm_path)
    diverged = diverged or llm_diverged
    llm_depth = len(llm_path)

    elapsed = time.perf_counter() - start
    return final_iri, path, diverged, bert_depth, llm_depth, elapsed


def run(tree, strategy_name):
    base_args = parse_args()
    model = base_args.model or DEFAULT_MODELS[base_args.backend]

    with open(base_args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Data is a dict keyed by entity IRI -> entity record.
    entities = list(raw.items())
    if base_args.limit:
        entities = entities[: base_args.limit]

    print(f"Strategy: {strategy_name}")
    print(f"Loaded {len(entities)} entities from {base_args.input}")
    print(f"LLM fallback -> Backend: {base_args.backend} | Model: {model} | Delay: {base_args.delay}s\n")

    results = []
    total_start = time.perf_counter()

    for i, (entity_iri, entry) in enumerate(entities):
        text = entry["description"]
        local_name = entry.get("local_name", entity_iri)
        true_depth = entry.get("actual_depth")  # ground-truth depth, NOT entry["depth"]
        true_parents = entry.get("direct_parents", [])

        print(f"[{i+1}/{len(entities)}] {local_name[:60]}  (true actual_depth={true_depth})")

        final_iri, path, diverged, bert_depth, llm_depth, elapsed = entity_iri, [], False, 0, 0, 0.0
        for attempt in range(3):
            try:
                final_iri, path, diverged, bert_depth, llm_depth, elapsed = classify_entity(
                    base_args, model, tree, entity_iri, text, base_args.max_depth
                )
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = base_args.delay * (2 ** attempt) * 5  # 5s -> 10s -> 20s
                    print(f"    ⚠️  Rate limited. Waiting {wait:.0f}s before retry {attempt+1}/3 …")
                    time.sleep(wait)
                else:
                    raise
        else:
            print("    ❌ Failed after 3 retries, skipping.")

        result = {
            "index":               i,
            "entity_iri":          entity_iri,
            "local_name":          local_name,
            "true_actual_depth":   true_depth,
            "true_direct_parents": [p["iri"] for p in true_parents],
            "path":                path,
            "bert_depth":          bert_depth,
            "llm_depth":           llm_depth,
            "predicted_depth":     bert_depth + llm_depth,
            "diverged":            diverged,
            "final_class_iri":     final_iri,
            "final_class_label":   _iri_to_label(final_iri),
            "elapsed_s":           round(elapsed, 3),
        }
        results.append(result)
        print(f"    → final: {result['final_class_label']}  "
              f"(bert_depth={bert_depth}, llm_depth={llm_depth}, diverged={diverged}, {elapsed:.2f}s)\n")

        if i < len(entities) - 1:
            time.sleep(base_args.delay)

    total_elapsed = time.perf_counter() - total_start
    avg = total_elapsed / len(entities) if entities else 0

    summary = {
        "timestamp":       datetime.now().isoformat(),
        "strategy":        strategy_name,
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
