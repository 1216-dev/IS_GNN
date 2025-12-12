"""Orchestration entrypoint tying together data loading, personas, graph, GNN, and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from data_loading import load_acs, load_demographics, load_facilities, load_incidents, load_income, load_race_state
from gnn_model import train_gnn, train_graph_gps
from graph_builder import build_graph
from personas import base_personas, synthesize_persona_examples
from rag_eval import evaluate_evacuations, generate_gt_with_reflection


DATA_DIR = Path(".")
FILES = {
    "demo": DATA_DIR / "data" / "Census_Demographics_at_the_Neighborhood_Tabulation_Area_(NTA)_level_20251126.csv",
    "acs": DATA_DIR / "data" / "demo_2016acs5yr_nyc.xlsx",
    "incidents": DATA_DIR / "data" / "Emergency_Response_Incidents_20251126.csv",
    "facilities": DATA_DIR / "data" / "Facilities_Database_20251126.csv",
    "income": DATA_DIR / "data" / "Income_By_Type_Of_Income_And_AGI_Range_20251127.csv",
    "race": DATA_DIR / "data" / "State.csv",
}


def _sample_graph(data, node_types, max_nodes: int = 80, max_edges: int = 120):
    """Return a small subgraph for visualization."""
    edge_index = data.edge_index.numpy()
    total_edges = edge_index.shape[1]
    edges = []
    nodes_set = set()
    # Deterministic sampling: take edges in order up to max_edges
    for idx in range(min(max_edges, total_edges)):
        src = int(edge_index[0, idx])
        dst = int(edge_index[1, idx])
        nodes_set.update([src, dst])
        edges.append({"source": src, "target": dst})
        if len(nodes_set) >= max_nodes:
            break
    nodes = []
    for nid in sorted(list(nodes_set))[:max_nodes]:
        ntype = node_types[nid] if nid < len(node_types) else "node"
        nodes.append({"id": nid, "type": ntype, "value": float(data.x[nid].mean().item())})
    return {"nodes": nodes, "edges": edges}


def run_pipeline(
    return_meta: bool = False,
    include_graph: bool = True,
    questions_per_persona: int = 80,
    sage_epochs: int = 4,
    transformer_epochs: int = 3,
    top_k: int = 5,
    seed: int = 42,
    return_examples: bool = False,
    research_mode: bool = False,
):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    if hasattr(np.random, "default_rng"):
        _ = np.random.default_rng(seed)

    demo = load_demographics(FILES["demo"])
    acs = load_acs(FILES["acs"])  # noqa: F841 (loaded for future feature engineering)
    incidents = load_incidents(FILES["incidents"])
    facilities = load_facilities(FILES["facilities"])
    income = load_income(FILES["income"])
    race = load_race_state(FILES["race"])

    borough_stats = _borough_stats(facilities, incidents, income, race)
    ctx_tags = _context_tags(demo, income, race)

    personas = base_personas()
    examples = []
    for p in personas:
        examples.extend(
            _examples_with_context(
                synthesize_persona_examples(
                    p,
                    demo,
                    incidents,
                    facilities,
                    n=questions_per_persona,
                    borough_stats=borough_stats,
                ),
                ctx_tags,
            )
        )


    graph_data, _, node_types, facility_map, incident_map, pe, edge_types = build_graph(demo, incidents, facilities)
    gnn_embs = train_gnn(graph_data, epochs=sage_epochs)
    transformer_embs = train_graph_gps(
        graph_data,
        epochs=transformer_epochs,
        edge_attr=graph_data.edge_attr,
        pos_enc=pe if research_mode else None,
        research_mode=research_mode,
    )

    # Split examples into 'Evacuation' (Domain Y) and others (Domain X)
    evac_examples = [ex for ex in examples if ex["persona"] == "Emergency Manager"]
    other_examples = [ex for ex in examples if ex["persona"] != "Emergency Manager"]
    
    # We evaluate primarily on the Evacuation examples as requested
    results_evac = evaluate_evacuations(
        evac_examples,
        gnn_embs,
        gps_embeddings=transformer_embs,
        top_k=top_k,
        facility_map=facility_map,
        incident_map=incident_map,
    )
    
    # Optional: Evaluate on others for comparison
    results_others = evaluate_evacuations(
        other_examples,
        gnn_embs,
        gps_embeddings=transformer_embs,
        top_k=top_k,
        facility_map=facility_map,
        incident_map=incident_map,
    )

    meta = {
        "personas": len(personas),
        "questions_per_persona": questions_per_persona,
        "total_examples": len(examples),
        "graph_nodes": int(graph_data.num_nodes),
        "graph_edges": int(graph_data.num_edges),
        "embedding_dim": int(gnn_embs.shape[1]),
        "datasets": [str(p.name) for p in FILES.values()],
        "context_tags": ctx_tags,
        "sage_epochs": sage_epochs,
        "transformer_epochs": transformer_epochs,
        "top_k": top_k,
        "seed": seed,
        "research_mode": research_mode,
    }
    # Return Evacuation results as the primary 'results' but include others
    results = {"evacuation": results_evac}
    results["other_domains_metrics"] = {k: v["metrics"] for k,v in results_others.items()}
    
    payload = {"results": results, "meta": meta}

    if return_examples:
        payload["examples"] = examples
        payload["gt_examples"] = generate_gt_with_reflection(examples)
    if include_graph:
        payload["graph_sample"] = _sample_graph(graph_data, node_types)
    if return_meta:
        return payload
    return results


def _context_tags(demo, income, race) -> Dict[str, str]:
    """Capture succinct tags so questions reference income/race/education context."""
    def _safe_top(df, col_candidates: List[str]) -> str:
        for col in col_candidates:
            if col in df.columns and not df.empty:
                return str(df[col].iloc[0])
        return "n/a"

    income_tag = _safe_top(income, ["income_group", "income_group", "geoname"])
    race_tag = _safe_top(race, ["lntitle", "geoname"])

    edu_cols = [c for c in demo.columns if "educ" in c or "school" in c or "bachelor" in c]
    education_tag = edu_cols[0] if edu_cols else "education_unknown"

    return {"income_tag": income_tag, "race_tag": race_tag, "education_tag": education_tag}


def _examples_with_context(examples, ctx_tags: Dict[str, str]):
    for ex in examples:
        income_tag = ex.get("income_tag", ctx_tags.get("income_tag", "income_unknown"))
        race_tag = ex.get("race_tag", ctx_tags.get("race_tag", "race_unknown"))
        edu_tag = ex.get("education_tag", ctx_tags.get("education_tag", "education_unknown"))
        ex["income_tag"] = income_tag
        ex["race_tag"] = race_tag
        ex["education_tag"] = edu_tag
        ex["question"] = (
            f"{ex['question']} (income:{income_tag} race:{race_tag} edu:{edu_tag})"
        )
    return examples


def _borough_stats(facilities, incidents, income_df, race_df):
    import random

    boroughs = set()
    if "boro" in facilities.columns:
        boroughs.update(facilities["boro"].dropna().unique().tolist())
    if "borough" in incidents.columns:
        boroughs.update(incidents["borough"].dropna().unique().tolist())
    income_groups = income_df["income_group"].dropna().tolist() if "income_group" in income_df.columns else ["median"]
    race_groups = race_df["geoname"].dropna().tolist() if "geoname" in race_df.columns else ["All"]
    stats = {}
    for b in boroughs:
        income_choice = random.choice(income_groups)
        race_choice = random.choice(race_groups)
        edu_score = round(random.uniform(0.2, 0.9), 2)
        stats[b] = {
            "income_tag": str(income_choice),
            "race_tag": str(race_choice),
            "education_tag": f"edu_score:{edu_score}",
        }
    return stats


if __name__ == "__main__":

    res = run_pipeline()
    print("METRICS_JSON_START")
    import json
    # Helper to convert numpy/torch to float for json dump
    def make_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    # Extract metrics only
    metrics_summary = {
        "evacuation": {k: v["metrics"] for k, v in res.get("evacuation", {}).items() if "metrics" in v},
        "other_domains": res.get("other_domains_metrics", {})
    }
    print(json.dumps(metrics_summary, default=make_serializable, indent=2))
    print("METRICS_JSON_END")

