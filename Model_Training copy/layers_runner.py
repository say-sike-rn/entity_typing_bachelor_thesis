### Uses every BERT layer model (4 total) to classify an entity and then makes an LLM choose from the BERT predictions. 
#!/usr/bin/env python3
"""
layers_runner.py
--------------------

Per entity:
  1. Run the BERT models layer1 to layer4 saving each result
  2. Ask the LLM to choose from the BERT predictions
  
  There is no loop here! Since the prompt already tells the LLM to choose the most specific class from the options given, we get the final class / direct parent of an entity immeadiately.
  Before calling the LLM, we check if any of the BERT predictions are a direct parent of the entity. 
"""

import argparse
import json
import re
import time
from datetime import datetime

LAYER_NAMES = ["layer1_model", "layer2_model", "layer3_model", "layer4_model"]

from LLM_2 import (
    DEFAULT_MODELS,
    _iri_to_label,
    build_prompt,
    call_llm,
    clean_label,
    is_entity_under_class,
)

from bert_cascade import (
    classify_with_bert,
    label_to_iri,
)

def parse_args():
    p = argparse.ArgumentParser(description="Bert model for each layer + LLM to choose from the BERT predictions")
    p.add_argument("--input", required=True, help="Path to input JSON file.")
    p.add_argument("--output", default="results.json", help="Path to write results JSON.")
    p.add_argument("--layers-dir", default="./models/layers",
                   help="Directory containing layer1_model .. layer4_model subfolders.")
    p.add_argument("--backend", default="mistral", help="LLM backend for the fallback stage.")
    p.add_argument("--model", default=None, help="Model name (backend-specific).")
    p.add_argument("--endpoint", default="https://untrained-upchuck-labrador.ngrok-free.dev", help="SPARQL endpoint.")
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
    (no SPARQL fetch) -- used to choose among the 4 BERT layer predictions.

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

    print("⏳ Calling LLM…")
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


def classify_entity(base_args, model, layer_dirs, entity_iri, text):
    """
    Full pipeline for one entity:
      1. Run the BERT layer models on text
      2. Run the LLM once to choose from the 4 predictions or say that it is none of them (FINISHED)

    Returns (final_iri, path, diverged, elapsed).
    """
    start = time.perf_counter()
    path = []
    diverged = False

    # BERT predictions
    bert_predictions = []
    for i, layer_dir in enumerate(layer_dirs):
        label, confidence, _all_scores = classify_with_bert(layer_dir, text, device=base_args.device)
        iri = label_to_iri(label)

        try:
            on_path = is_entity_under_class(base_args.endpoint, entity_iri, iri)
        except Exception as e:
            print(f"    ⚠️  Divergence check failed ({e}); assuming on-path.")
            on_path = True

        path.append({
            "stage": f"bert_layer{i + 1}",
            "step": i + 1,
            "iri": iri,
            "label": label,
            "confidence": round(confidence, 4),
            "model_dir": layer_dir,
            "on_path": on_path,
        })
        bert_predictions.append({"iri": iri, "label": label, "on_path": on_path})

        print(f"    → layer{i + 1} predicted: {label}  (on_path={on_path})")

    # If a BERT layer doesn't find a valid path, skip the LLM call entirely.
    direct_parent_hit = next((p for p in bert_predictions if p["on_path"]), None)
    # "on_path" here only tells us the prediction is an ANCESTOR of the
    # entity's true type via SPARQL -- not specifically that it's a literal direct parent. 
    if not direct_parent_hit:
        print(f"    BERT failed to find a valid path for {entity_iri} -- skipping LLM call.")
        elapsed = time.perf_counter() - start
        return "something went wrong rahhhh", path, diverged, elapsed

    fixed_subclasses = [{"iri": p["iri"], "label": p["label"]} for p in bert_predictions]
    current_label = "Person" 

    fixed_result = run_fixed_subclass_step(base_args, model, current_label, fixed_subclasses, text)

    if fixed_result in ("something went wrong rahhhh", "FINISHED"):
        elapsed = time.perf_counter() - start
        return fixed_result, path, diverged, elapsed

    step_label = _iri_to_label(fixed_result)
    try:
        on_path = is_entity_under_class(base_args.endpoint, entity_iri, fixed_result)
    except Exception as e:
        print(f"    ⚠️  Divergence check failed ({e}); assuming on-path.")
        on_path = True

    path.append({
        "stage": "llm_choose_layer",
        "step": 1,
        "iri": fixed_result,
        "label": step_label,
        "on_path": on_path,
    })

    if not on_path:
        diverged = True
        print(f"    ⚠️  Diverged at LLM choice: "
              f"'{step_label}' is not an ancestor of the entity's true type(s).")

    elapsed = time.perf_counter() - start
    return fixed_result, path, diverged, elapsed



def run():
    base_args = parse_args()
    model = base_args.model or DEFAULT_MODELS[base_args.backend]

    with open(base_args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entities = list(raw.items())
    if base_args.limit:
        entities = entities[: base_args.limit]

    layer_dirs = [f"{base_args.layers_dir}/{name}" for name in LAYER_NAMES]

    print("Strategy: layers")
    print(f"Loaded {len(entities)} entities from {base_args.input}")
    print(f"Layer model: {layer_dirs}")
    print(f"LLM fallback -> Backend: {base_args.backend} | Model: {model} | Delay: {base_args.delay}s\n")

    results = []
    total_start = time.perf_counter()

    for i, (entity_iri, entry) in enumerate(entities):
        text = entry["description"]
        local_name = entry.get("local_name", entity_iri)
        true_depth = entry.get("actual_depth")
        true_parents = entry.get("direct_parents", [])

        print(f"[{i+1}/{len(entities)}] {local_name[:60]}  (true actual_depth={true_depth})")

        final_iri, path, diverged, elapsed = entity_iri, [], False, 0.0
        for attempt in range(10):
            try:
                final_iri, path, diverged, elapsed = classify_entity(
                    base_args, model, layer_dirs, entity_iri, text
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
            "path":                path,
            "predicted_depth":     len(path),
            "diverged":            diverged,
            "final_class_iri":     final_iri,
            "final_class_label":   _iri_to_label(final_iri),
            "elapsed_s":           round(elapsed, 3),
        }

        results.append(result)
        print(f"    → final: {result['final_class_label']}  "
              f"(diverged={diverged}, {elapsed:.2f}s)\n")

        if i < len(entities) - 1:
            time.sleep(base_args.delay)

    total_elapsed = time.perf_counter() - total_start
    avg = total_elapsed / len(entities) if entities else 0

    summary = {
        "timestamp":       datetime.now().isoformat(),
        "strategy":        "layers",
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