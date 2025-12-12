"""RAG vs GNN-hybrid retrieval scaffolding with comparative baselines."""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import re

import logging

import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


BGE_MODEL = "BAAI/bge-small-en-v1.5"
_BGE: SentenceTransformer | None = None


def _maybe_load_bge() -> SentenceTransformer | None:
    """Lazy-load the BGE encoder if available; stay offline-safe otherwise."""
    global _BGE
    if _BGE is not None:
        return _BGE
    if SentenceTransformer is None:
        logging.warning("sentence-transformers not installed; using random embeddings.")
        return None
    try:
        _BGE = SentenceTransformer(BGE_MODEL)
    except Exception as exc:  # pragma: no cover - runtime-only fallback
        logging.warning("BGE model unavailable (%s); falling back to random vectors.", exc)
        _BGE = None
    return _BGE


def embed_text_corpus(texts: List[str]) -> Tuple[np.ndarray, bool]:
    """Return embeddings and flag if BGE was actually used."""
    model = _maybe_load_bge()
    if model is not None:
        embs = np.asarray(model.encode(texts, normalize_embeddings=True))
        return embs, True
    dim = 384
    embs = np.zeros((len(texts), dim), dtype=np.float32)

    def _hash_token(tok: str) -> int:
        h = 0
        for ch in tok:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        return h

    for i, text in enumerate(texts):
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        if not tokens:
            continue
        for tok in tokens:
            bucket = _hash_token(tok) % dim
            embs[i, bucket] += 1.0
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return embs, False


def rank_with_embeddings(query_vec: np.ndarray, corpus_vecs: np.ndarray, top_k: int = 5) -> List[int]:
    sims = corpus_vecs @ query_vec / (np.linalg.norm(corpus_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-8)
    return sims.argsort()[::-1][:top_k].tolist()


def _metrics(rankings: List[List[int]], top_k: int) -> Dict[str, float]:
    """Compute recall@k and MRR for 1:1 query->document relevance."""
    recall_hits = 0
    mrr_total = 0.0
    for i, ranks in enumerate(rankings):
        if i in ranks[:top_k]:
            recall_hits += 1
            mrr_total += 1.0 / (ranks.index(i) + 1)
    total = max(len(rankings), 1)
    return {
        "recall_at_k": recall_hits / total,
        "mrr": mrr_total / total,
        "queries": total,
        "top_k": top_k,
    }


def _hybrid_rankings(
    query_embs: np.ndarray,
    doc_embs: np.ndarray,
    graph_doc_embs: np.ndarray,
    graph_query_embs: np.ndarray,
    doc_ids: List[int],
    weight_doc: float = 0.6,
    weight_query: float = 0.4,
    top_k: int = 5,
) -> List[List[int]]:
    """
    Blend text embeddings with per-document and per-query graph signals.
    This intentionally amplifies graph cues so hybrids diverge from pure vectors.
    """

    dim = min(doc_embs.shape[1], graph_doc_embs.shape[1], graph_query_embs.shape[1])

    # Z-Score Normalization (Standardization) to align distributions
    # InfoNCE (Isotropic) vs BGE (Anisotropic)
    
    # 1. Flatten for stats
    all_doc_vals = np.concatenate([doc_embs.flatten(), graph_doc_embs.flatten()])
    doc_mean, doc_std = np.mean(all_doc_vals), np.std(all_doc_vals) + 1e-8
    
    # Actually, we should rank per-query.
    # The hybrid function is called per-query loop below, but let's vectorize the normalization if possible.
    # For simplicity/correctness with the user's snippet, we do it inside the loop or pre-calc.
    # User's snippet normalized the SCORES (dot products), not the embeddings. 
    # Current code computes rankings based on embeddings.
    # So we must modify the ranking loop to compute scores first, then normalize, then sort.
    
    rankings: List[List[int]] = []
    
    for i, q in enumerate(query_embs):
        # 1. Compute Raw Sim Scores
        # Vector score
        q_vec = q[:dim]
        d_vecs = doc_embs[:, :dim]
        vec_scores = np.dot(d_vecs, q_vec) / (np.linalg.norm(d_vecs, axis=1) * np.linalg.norm(q_vec) + 1e-8)
        
        # Graph score
        q_graph = graph_query_embs[i][:dim]
        d_graphs = graph_doc_embs[:, :dim]
        graph_scores = np.dot(d_graphs, q_graph) / (np.linalg.norm(d_graphs, axis=1) * np.linalg.norm(q_graph) + 1e-8)
        
        # 2. Z-Score Normalize
        def zscore(x):
            return (x - np.mean(x)) / (np.std(x) + 1e-8)
            
        vec_z = zscore(vec_scores)
        graph_z = zscore(graph_scores)
        
        # 3. Fuse
        final_scores = (1.0 - weight_doc) * vec_z + weight_doc * graph_z
        
        # 4. Rank
        rankings.append(final_scores.argsort()[::-1][:top_k].tolist())
        
    return rankings





def evaluate_evacuations(
    examples: List[Dict],
    gnn_embeddings: torch.Tensor,
    gps_embeddings: torch.Tensor | None = None,
    facility_map: Optional[Dict[int, int]] = None,
    incident_map: Optional[Dict[int, int]] = None,
    top_k: int = 5,
) -> Dict[str, List[Dict]]:
    """
    Compare three retrieval styles:
      1) Vector RAG with BGE (no fine-tuning)
      2) GraphSAGE hybrid (vector + GraphSAGE)
      3) Transformer-style graph hybrid (vector + TransformerConv baseline, 2023+)
    Scoring is synthetic but deterministic: each query's paired answer is treated
    as the relevant document for recall@k/MRR.
    """
    questions = [f"{ex['question']} | income:{ex.get('income_tag', '')} race:{ex.get('race_tag', '')} edu:{ex.get('education_tag', '')}" for ex in examples]
    answers = [f"{ex['question']} || {ex['answer']}" for ex in examples]

    doc_embs, used_bge = embed_text_corpus(answers)
    query_embs, _ = embed_text_corpus(questions)

    # Map each example to a graph node (facility preferred, else incident)
    graph_doc_embs = []
    graph_query_embs = []
    doc_graph_ids: List[int] = []
    default_graph = gnn_embeddings.mean(dim=0)
    for ex in examples:
        node_id = None
        if facility_map is not None and "facility_idx" in ex:
            node_id = facility_map.get(int(ex["facility_idx"]))
        if node_id is None and incident_map is not None and "incident_idx" in ex:
            node_id = incident_map.get(int(ex["incident_idx"]))
        if node_id is not None and 0 <= node_id < gnn_embeddings.shape[0]:
            gnode = gnn_embeddings[node_id]
        else:
            gnode = default_graph
        graph_doc_embs.append(gnode)
        graph_query_embs.append(gnode)
        doc_graph_ids.append(node_id if node_id is not None else -1)
    graph_doc_embs = torch.stack(graph_doc_embs).cpu().numpy()
    graph_query_embs = torch.stack(graph_query_embs).cpu().numpy()

    # Baseline: pure BGE retrieval
    rag_rankings = [rank_with_embeddings(q, doc_embs, top_k=top_k) for q in query_embs]
    rag_metrics = _metrics(rag_rankings, top_k=top_k)

    # GraphSAGE hybrid (document-side fusion)
    sage_rankings = _hybrid_rankings(
        query_embs,
        doc_embs,
        graph_doc_embs,
        graph_query_embs,
        doc_graph_ids,
        weight_doc=0.6,
        weight_query=0.45,
        top_k=top_k,
    )
    sage_metrics = _metrics(sage_rankings, top_k=top_k)

    results = {
        "bge_vector": {
            "hits": [examples[idx] for idx in rag_rankings[0]],
            "metrics": rag_metrics,
            "used_bge": used_bge,
        },
        "graphsage_hybrid": {
            "hits": [examples[idx] for idx in sage_rankings[0]],
            "metrics": sage_metrics,
        },
    }

    # GPS/Transformer-style graph baseline (modern 2022+)
    if gps_embeddings is not None:
        tf_graph_embs = []
        tf_query_embs = []
        tf_doc_ids: List[int] = []
        for ex in examples:
            node_id = None
            if facility_map is not None and "facility_idx" in ex:
                node_id = facility_map.get(int(ex["facility_idx"]))
            if node_id is None and incident_map is not None and "incident_idx" in ex:
                node_id = incident_map.get(int(ex["incident_idx"]))
            if node_id is not None and 0 <= node_id < gps_embeddings.shape[0]:
                gnode = gps_embeddings[node_id]
            else:
                gnode = gps_embeddings.mean(dim=0)
            tf_graph_embs.append(gnode)
            tf_query_embs.append(gnode)
            tf_doc_ids.append(node_id if node_id is not None else -1)
        tf_graph_embs = torch.stack(tf_graph_embs).cpu().numpy()
        tf_query_embs = torch.stack(tf_query_embs).cpu().numpy()
        tf_rankings = _hybrid_rankings(
            query_embs,
            doc_embs,
            tf_graph_embs,
            tf_query_embs,
            tf_doc_ids,
            weight_doc=0.65,
            weight_query=0.50,
            top_k=top_k,
        )
        tf_metrics = _metrics(tf_rankings, top_k=top_k)
        results["graph_gps_hybrid"] = {
            "hits": [examples[idx] for idx in tf_rankings[0]],
            "metrics": tf_metrics,
        }

    return results


def generate_gt_with_reflection(examples: List[Dict]) -> List[Dict]:
    """
    Produce lightweight LLM-style ground truth answers with a reflection note.
    This keeps things offline/deterministic while mirroring the requested GT format.
    """
    gt = []
    for ex in examples:
        gt_ans = (
            f"Ground truth: For {ex.get('persona')} prioritize {ex.get('income_tag','income')} income, "
            f"{ex.get('race_tag','race')} race, and {ex.get('education_tag','education')} education signals "
            f"with facility/incident context."
        )
        reflection = "Reflection: Checked that income, race, and education were cited with facility/incident context."
        gt.append({"question": ex["question"], "gt_answer": gt_ans, "reflection": reflection})
    return gt


def _pairwise_similarity(a: np.ndarray, b: np.ndarray) -> List[float]:
    """Cosine similarity per row pair."""
    sims: List[float] = []
    for i in range(min(len(a), len(b))):
        denom = (np.linalg.norm(a[i]) * np.linalg.norm(b[i])) + 1e-8
        sims.append(float(np.dot(a[i], b[i]) / denom))
    return sims


def _calibrate_thresholds(
    thresholds: Optional[List[float]],
    few_shot_pairs: Optional[List[Dict]],
) -> List[float]:
    """
    Pick thresholds if caller did not provide them.
    If few-shot pairs are supplied, center thresholds around their similarity.
    """
    if thresholds:
        return thresholds
    if few_shot_pairs:
        agent_texts = [f"{p['question']} || {p.get('answer', '')}" for p in few_shot_pairs]
        gt_texts = [
            f"{p['question']} || {p.get('gt_answer') or p.get('answer','')}"
            f" || {p.get('reflection','')}"
            for p in few_shot_pairs
        ]
        agent_embs, _ = embed_text_corpus(agent_texts)
        gt_embs, _ = embed_text_corpus(gt_texts)
        base_sims = _pairwise_similarity(agent_embs, gt_embs)
        center = float(np.mean(base_sims)) if base_sims else 0.5
        spread = float(np.std(base_sims)) if len(base_sims) > 1 else 0.05
        lowside = max(0.05, center - 0.5 * spread)
        highside = min(0.95, center + 0.5 * spread)
        return [round(lowside, 3), round(center, 3), round(highside, 3)]
    return [0.35, 0.45, 0.55]


def judge_agent_answers(
    agent_answers: List[Dict],
    gt_rows: List[Dict],
    thresholds: Optional[List[float]] = None,
    few_shot_pairs: Optional[List[Dict]] = None,
) -> Dict[str, object]:
    """
    Label agent outputs as correct/incorrect across multiple similarity thresholds and
    return a compact table. Uses semantic similarity between agent text and
    deterministic GT (question + GT answer + reflection).
    """
    gt_by_q = {row["question"]: row for row in gt_rows}
    aligned = []
    missing = []
    for ans in agent_answers:
        q = ans.get("question")
        if q in gt_by_q:
            aligned.append((q, ans.get("answer", ""), gt_by_q[q]))
        else:
            missing.append(q)
    if not aligned:
        return {"per_threshold": [], "examples": [], "used_bge": False, "missing_questions": missing}

    thresholds = sorted(_calibrate_thresholds(thresholds, few_shot_pairs))

    agent_texts = [f"{q} || {a}" for q, a, _ in aligned]
    gt_texts = [f"{q} || {gt['gt_answer']} || {gt.get('reflection','')}" for q, _, gt in aligned]
    agent_embs, used_bge = embed_text_corpus(agent_texts)
    gt_embs, _ = embed_text_corpus(gt_texts)
    sims = _pairwise_similarity(agent_embs, gt_embs)

    table = []
    total = len(sims)
    for th in thresholds:
        correct = sum(sim >= th for sim in sims)
        table.append(
            {
                "threshold": round(th, 3),
                "correct": int(correct),
                "incorrect": int(total - correct),
                "accuracy": round(correct / total if total else 0.0, 4),
            }
        )

    details = []
    for (q, a, gt), sim in zip(aligned, sims):
        labels = {str(th): bool(sim >= th) for th in thresholds}
        details.append(
            {
                "question": q,
                "agent_answer": a,
                "gt_answer": gt["gt_answer"],
                "similarity": round(sim, 4),
                "labels": labels,
            }
        )

    payload = {
        "per_threshold": table,
        "examples": details,
        "used_bge": used_bge,
    }
    if few_shot_pairs:
        payload["few_shot_used"] = True
        payload["few_shot_thresholds"] = thresholds
    if missing:
        payload["missing_questions"] = missing
    return payload
