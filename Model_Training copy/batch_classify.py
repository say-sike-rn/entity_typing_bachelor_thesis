#!/usr/bin/env python3
"""
batch_classify.py
-----------------
Iterates over entities (keyed by IRI) from a JSON file, runs LLM-only.loop()
classification for each, tracks the path taken through the ontology, and
stops early if the LLM diverges from a class that's actually an ancestor
of one of the entity's true types.

Usage
-----
python batch_classify.py \
    --input entities.json \
    --backend mistral \
    --endpoint "http://localhost:9004" \
    --output results.json \
    --delay 2.0
"""

import argparse
import json
import time
from datetime import datetime

from LLM-only import loop, _iri_to_label, DEFAULT_MODELS, is_entity_under_class


def parse_args():
    p = argparse.ArgumentParser(description="Batch entity classification with timing.")
    p.add_argument("--input",    required=True, help="Path to input JSON file.")
    p.add_argument("--output",   default="results.json", help="Path to write results JSON.")
    p.add_argument("--backend",  default="mistral", help="LLM backend.")
    p.add_argument("--model",    default=None,      help="Model name (backend-specific).")
    p.add_argument("--endpoint", default="http://localhost:9004", help="SPARQL endpoint.")
    p.add_argument("--iri",      default="http://schema.org/Person", help="Starting class IRI.")
    p.add_argument("--api-key",  default=None)
    p.add_argument("--api-base", default="https://api.openai.com/v1")
    p.add_argument("--subclass-limit", type=int, default=1000)
    p.add_argument("--entity-sample",  type=int, default=10)
    p.add_argument("--temperature",    type=float, default=0.0)
    p.add_argument("--max-tokens",     type=int, default=128)
    p.add_argument("--max-depth",      type=int, default=10, help="Max ontology traversal depth.")
    p.add_argument("--delay",          type=float, default=1.0,
                   help="Seconds to wait between entities (avoid rate limits). Default: 1.0")
    p.add_argument("--limit",          type=int, default=6,
                   help="Max number of entities to process from the input file.")
    p.add_argument("--dry-run",        action="store_true")
    return p.parse_args()


class Args:
    """Mutable args object passed into loop() for each entity."""
    def __init__(self, base_args, description):
        self.iri            = base_args.iri
        self.entity         = None
        self.description    = description
        self.endpoint       = base_args.endpoint
        self.backend        = base_args.backend
        self.model          = base_args.model
        self.api_base       = base_args.api_base
        self.api_key        = base_args.api_key
        self.subclass_limit = base_args.subclass_limit
        self.entity_sample  = base_args.entity_sample
        self.temperature    = base_args.temperature
        self.max_tokens     = base_args.max_tokens
        self.dry_run        = base_args.dry_run


def classify_entity(base_args, model, entity_iri, text, max_depth):
    """
    Run the classification loop for a single entity.

    Returns
    -------
    final_iri : str
        The last class the LLM actually chose (even if it diverged there).
    path : list[dict]
        Every step taken: {"step", "iri", "label", "on_path"}.
    diverged : bool
        True if the run was stopped because the chosen class was not an
        ancestor of any of the entity's real types.
    depth : int
        Number of *valid* (on-path) steps taken.
    elapsed : float
        Wall-clock seconds spent on this entity.
    """
    args = Args(base_args, description=text)
    current_label = _iri_to_label(args.iri)
    depth = 0
    path = []
    diverged = False
    last_result = args.iri  # fallback if the very first call fails

    start = time.perf_counter()

    for _ in range(max_depth):
        result = loop(args, model, current_label)

        if result in ("something went wrong rahhhh", "FINISHED"):
            break

        last_result = result
        step_label = _iri_to_label(result)

        # Check whether the chosen class is still on a valid path, i.e. an
        # ancestor (or the class itself) of one of the entity's true rdf:types.
        try:
            on_path = is_entity_under_class(base_args.endpoint, entity_iri, result)
        except Exception as e:
            print(f"    ⚠️  Divergence check failed ({e}); assuming on-path.")
            on_path = True

        path.append({
            "step":    depth + 1,
            "iri":     result,
            "label":   step_label,
            "on_path": on_path,
        })

        if not on_path:
            diverged = True
            print(f"    ⚠️  Diverged at step {depth + 1}: '{step_label}' is not an ancestor of the entity's true type(s).")
            break

        args.iri = result
        current_label = step_label
        depth += 1

    elapsed = time.perf_counter() - start
    return last_result, path, diverged, depth, elapsed


def main():
    base_args = parse_args()
    model = base_args.model or DEFAULT_MODELS[base_args.backend]

    with open(base_args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Data is a dict keyed by entity IRI -> entity record.
    entities = list(raw.items())[: base_args.limit]

    print(f"Loaded {len(entities)} entities from {base_args.input}")
    print(f"Backend: {base_args.backend} | Model: {model} | Delay: {base_args.delay}s\n")

    results = []
    total_start = time.perf_counter()

    for i, (entity_iri, entry) in enumerate(entities):
        text        = entry["description"]
        local_name  = entry.get("local_name", entity_iri)
        true_depth  = entry.get("actual_depth")
        true_parents = entry.get("direct_parents", [])

        print(f"[{i+1}/{len(entities)}] {local_name[:60]}  (true actual_depth={true_depth})")

        final_iri, path, diverged, depth, elapsed = base_args.iri, [], False, 0, 0.0
        skipped = False
        last_error = None
        for attempt in range(3):
            try:
                final_iri, path, diverged, depth, elapsed = classify_entity(
                    base_args, model, entity_iri, text, base_args.max_depth
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
            skipped = True
            print(f"    ❌ Failed after 3 retries, skipping. Last error: {last_error[:80]}")

        result = {
            "index":               i,
            "entity_iri":          entity_iri,
            "local_name":          local_name,
            "true_actual_depth":   true_depth,
            "true_direct_parents": [p["iri"] for p in true_parents],
            "path":                path,
            "predicted_depth":     depth,
            "diverged":            diverged,
            "final_class_iri":     final_iri if not skipped else None,
            "final_class_label":   _iri_to_label(final_iri) if not skipped else None,
            "elapsed_s":           round(elapsed, 3),
            "skipped":             skipped,
            "error":               last_error if skipped else None,
        }
        
        results.append(result)
        print(f"    → final: {result['final_class_label']}  (predicted_depth={depth}, diverged={diverged}, {elapsed:.2f}s)\n")
        if i < len(entities) - 1:
            time.sleep(base_args.delay)

    total_elapsed = time.perf_counter() - total_start
    avg = total_elapsed / len(entities) if entities else 0

    summary = {
        "timestamp":       datetime.now().isoformat(),
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
    main()