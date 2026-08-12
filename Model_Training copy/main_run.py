#!/usr/bin/env python3
"""
run_standard.py
----------------
"Standard" strategy: classify from Person using BERT. If the top-level model
predicts Engineer or Competitive_Player, descend into that sub-model. Once
the cascade reaches a label with no further BERT model, hand off to the LLM
(see strategy_runner.py for the shared driver).

Usage
-----
python run_standard.py --input sampled_1000.json --backend mistral --output results_standard.json
"""

import os

from bert_cascade import CascadeNode
from standard_runner import run
from bertopics_runner import run as run_topics

MODELS_STANDARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "standard")
MODELS_TOPICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "topics")

STANDARD_TREE = CascadeNode(
    model_dir=os.path.join(MODELS_STANDARD_DIR, "person_model"),
    next={
        "Engineer": CascadeNode(model_dir=os.path.join(MODELS_STANDARD_DIR, "engineer_model")),
        "Competitive_Player_Q18536342": CascadeNode(model_dir=os.path.join(MODELS_STANDARD_DIR, "competitive_player_model")),
    },
)

if __name__ == "__main__":
    run(STANDARD_TREE, strategy_name="standard")
