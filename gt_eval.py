"""
Utility to generate LLM-style ground truth with reflection and run a small evaluation.
Run: python3 gt_eval.py
"""

from __future__ import annotations

from typing import Dict

from pipeline import run_pipeline
from rag_eval import generate_gt_with_reflection, judge_agent_answers


def main():
    # Smaller sample to keep quick/offline.
    payload = run_pipeline(
        return_meta=True,
        include_graph=False,
        questions_per_persona=20,
        sage_epochs=2,
        transformer_epochs=2,
        top_k=5,
        seed=123,
        return_examples=True,
    )
    examples = payload["examples"]
    meta = payload["meta"]

    # Generate deterministic GT with reflection note.
    gt = generate_gt_with_reflection(examples)

    # Simulate agent answers and score correctness across thresholds.
    agent_answers = []
    for idx, ex in enumerate(examples[:30]):
        agent_ans = ex["answer"]
        if idx % 7 == 0:
            agent_ans = "Generic response without demographic grounding."
        agent_answers.append({"question": ex["question"], "answer": agent_ans})

    judgement = judge_agent_answers(agent_answers, gt, thresholds=[0.35, 0.5, 0.65])

    summary: Dict[str, Dict] = {"meta": meta, "gt_examples": gt[:5]}
    print("GT preview (first 5):")
    for row in summary["gt_examples"]:
        print("-", row["question"])
        print("  ", row["gt_answer"])
        print("  ", row["reflection"])
    print("\nCorrectness table (agent vs GT):")
    for row in judgement["per_threshold"]:
        print(
            f"thr={row['threshold']:.2f} | acc={row['accuracy']:.3f} | "
            f"correct={row['correct']} | incorrect={row['incorrect']}"
        )
    if "examples" in judgement:
        print("\nSample decisions:")
        for det in judgement["examples"][:4]:
            label_str = " ".join(f"{thr}:{int(flag)}" for thr, flag in det["labels"].items())
            print(f"- sim={det['similarity']:.3f} | {label_str} | Q: {det['question'][:90]}...")
    print(f"\nUsed BGE encoder: {judgement['used_bge']}")


if __name__ == "__main__":
    main()
