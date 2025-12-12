"""
Streamlit UI to run the comparative retrieval experiment (BGE vector, GraphSAGE
hybrid, Transformer-GNN hybrid) and visualize quick stats.
Run with: streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st
import networkx as nx
import plotly.graph_objects as go

from pipeline import run_pipeline
from rag_eval import generate_gt_with_reflection


def card_list(title: str, hits, style_class=""):
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)
    if not hits:
        st.caption("No hits found.")
        return

    for hit in hits:
        # Extract meta from question if available (some have parens)
        q_text = hit['question']
        meta_html = ""
        if "(" in q_text and ")" in q_text:
            # Simple hack to de-clutter if needed, or just leave it
            pass
            
        html = f"""
        <div class='card {style_class}'>
            <div class='persona'>{hit.get('persona', 'User')}</div>
            <div class='question'>{q_text}</div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


def graph_viz(graph_sample: dict):
    if not graph_sample:
        return None
    G = nx.Graph()
    node_types = {}
    for n in graph_sample.get("nodes", []):
        G.add_node(n["id"], value=n.get("value", 0), ntype=n.get("type", "node"))
        node_types[n["id"]] = n.get("type", "node")
    for e in graph_sample.get("edges", []):
        G.add_edge(e["source"], e["target"])
    pos = nx.spring_layout(G, seed=7, k=0.25)

    type_palette = {"neighborhood": "#22d3ee", "facility": "#f97316", "incident": "#a78bfa"}
    xs_lines = []
    ys_lines = []
    for src, dst in G.edges():
        xs_lines.extend([pos[src][0], pos[dst][0], None])
        ys_lines.extend([pos[src][1], pos[dst][1], None])
    edge_trace = go.Scatter(x=xs_lines, y=ys_lines, mode="lines", line=dict(color="rgba(255,255,255,0.15)", width=1), hoverinfo="none")

    node_traces = []
    for ntype, color in type_palette.items():
        nx_nodes = [n for n in G.nodes() if node_types.get(n) == ntype]
        if not nx_nodes:
            continue
        node_traces.append(
            go.Scatter(
                x=[pos[n][0] for n in nx_nodes],
                y=[pos[n][1] for n in nx_nodes],
                mode="markers",
                marker=dict(size=10, color=color, line=dict(width=1, color="rgba(255,255,255,0.6)")),
                name=ntype,
                hovertext=[f"{ntype} #{n}" for n in nx_nodes],
                hoverinfo="text",
            )
        )

    fig = go.Figure()
    fig.add_trace(edge_trace)
    for tr in node_traces:
        fig.add_trace(tr)
    fig.update_layout(
        showlegend=True,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig



def main():
    st.set_page_config(page_title="Persona-aware Evacuation Explorer", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for "Dashboard" feel
    st.markdown(
        """
        <style>
        body {background-color: #0e1117; color: #e2e8f0; font-family: sans-serif;}
        .stButton>button {width: 100%; border-radius: 6px; font-weight: 600;}
        
        /* HEADER */
        .header-container {
            background: linear-gradient(90deg, #38bdf8 0%, #3b82f6 100%);
            padding: 2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 2rem;
        }
        .header-title {font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem;}
        .header-subtitle {font-size: 1.1rem; opacity: 0.9; margin-bottom: 1.5rem;}
        .badges {display: flex; gap: 10px; flex-wrap: wrap;}
        .badge {
            background-color: rgba(255, 255, 255, 0.2);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            backdrop-filter: blur(4px);
        }

        /* METRICS GRID */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 10px;
        }
        .metric-box {
            background-color: #0f1116; /* Darker background */
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 15px;
        }
        .metric-label {
            color: #64748b;
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 5px;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #f1f5f9;
        }
        .context-line {
            font-family: monospace;
            color: #94a3b8;
            font-size: 0.9rem;
            margin-top: 5px;
            margin-bottom: 20px;
            border-left: 2px solid #3b82f6;
            padding-left: 10px;
        }

        /* CARDS */
        .section-header {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 10px;
            color: #e2e8f0;
        }
        .card {
            background-color: #1e293b;
            border-left: 4px solid #3b82f6;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .card.sage { border-left-color: #22d3ee; }
        .card.gps { border-left-color: #4ade80; } /* Greenish for GPS */
        
        .persona {
            color: #38bdf8;
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 4px;
        }
        .question {
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.4;
        }
        .meta {
            font-size: 0.8rem;
            color: #64748b;
            margin-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("Control Panel")
        if "seed" not in st.session_state:
            st.session_state.seed = 42
        
        st.session_state.seed = st.number_input("Random Seed", value=st.session_state.seed, min_value=0, max_value=9999)
        if st.button("Shuffle Seed"):
            import random
            st.session_state.seed = random.randint(0, 9999)
            st.rerun()
            
        st.divider()
        research_mode = st.toggle("Research Mode (GPS + positional + edge types)", value=True)
        st.info("Uses Semantic Edges & Z-Score Fusion when enabled.")
        
        st.divider()
        quick_run = st.button("Quick Run (Fast)", type="primary")
        full_run = st.button("Full Run (Accurate)")

    # Header
    st.markdown("""
    <div class="header-container">
        <div class="header-title">Persona-aware Evacuation Explorer</div>
        <div class="header-subtitle">Compare raw RAG vs. graph-informed retrieval with income, race, and education signals baked in.</div>
        <div class="badges">
            <div class="badge">Personas: Emergency/Planner/Health/Social/Transit/Utility/Equity</div>
            <div class="badge">Data: Demographics • ACS • Incidents • Facilities • Income • Race</div>
            <div class="badge">Models: BGE vector • GraphSAGE hybrid • Transformer-GNN hybrid</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if quick_run or full_run:
        with st.spinner("Running pipeline with Research Optimizations..."):
            # Set params based on mode
            q_per_p = 120 if full_run else 40
            epochs = 4 if full_run else 2
            
            payload = run_pipeline(
                return_meta=True,
                questions_per_persona=q_per_p,
                sage_epochs=epochs,
                transformer_epochs=epochs,
                top_k=5,
                seed=st.session_state.seed,
                return_examples=True,
                research_mode=research_mode,
            )
            
        results = payload["results"]
        meta = payload["meta"]
        m_evac = results.get("evacuation", {})

        # --- KEY METRICS ROW ---
        # --- METRICS GRID ---
        st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Personas</div>
                <div class="metric-value">{meta['personas']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Graph Nodes</div>
                <div class="metric-value">{meta['graph_nodes']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total Q&A</div>
                <div class="metric-value">{meta['total_examples']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Questions / Persona</div>
                <div class="metric-value">{meta['questions_per_persona']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Graph Edges</div>
                <div class="metric-value">{meta['graph_edges']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Embedding Dim</div>
                <div class="metric-value">{meta['embedding_dim']}</div>
            </div>
        </div>
        <div class="context-line">
            Context tags • income: {meta['context_tags']['income_tag']} • race: {meta['context_tags']['race_tag']} • edu: {meta['context_tags']['education_tag']}
        </div>
        """, unsafe_allow_html=True)
        
        # --- RETRIEVAL COLUMNS ---
        c1, c2, c3 = st.columns(3)
        with c1:
            card_list("BGE vector (frozen)", m_evac.get("bge_vector", {}).get("hits", []), style_class="")
        with c2:
            card_list("GraphSAGE hybrid", m_evac.get("graphsage_hybrid", {}).get("hits", []), style_class="sage")
        with c3:
            card_list("Transformer-GNN hybrid", m_evac.get("graph_gps_hybrid", {}).get("hits", []), style_class="gps")

        st.write("---")

        # Tabs for extra details
        tab1, tab2 = st.tabs(["📊 Evaluation Tables", "🔍 Overlap & GT"])
        
        with tab1:
            st.subheader("Evacuation Expert (Domain Y) - The Target")
            # Create a nice dataframe
            data = []
            for model, pretty in [("bge_vector", "BGE Vector"), ("graphsage_hybrid", "GraphSAGE"), ("graph_gps_hybrid", "GraphGPS")]:
                if model in m_evac:
                    metrics = m_evac[model]["metrics"]
                    data.append({
                        "Model Architecture": pretty,
                        "Recall@5": metrics.get("recall_at_k"),
                        "MRR": metrics.get("mrr"),
                        "Methodology": "Vector Only" if "vector" in model else "Z-Score Fusion"
                    })
            st.dataframe(data, use_container_width=True)
            
            if "other_domains_metrics" in results:
                st.subheader("Reference Domains (Domain X)")
                st.caption("Performance on general city planning queries (non-evacuation)")
                data_x = []
                m_other = results["other_domains_metrics"]
                for model, pretty in [("bge_vector", "BGE Vector"), ("graphsage_hybrid", "GraphSAGE"), ("graph_gps_hybrid", "GraphGPS")]:
                    if model in m_other:
                        metrics = m_other[model]
                        data_x.append({
                            "Model": pretty,
                            "Recall@5": metrics.get("recall_at_k"),
                            "MRR": metrics.get("mrr")
                        })
                st.dataframe(data_x, use_container_width=True)

        with tab2:
            st.subheader("Overlap signal")
            # Overlap calculation (using bge from m_evac)
            bge_hits = m_evac.get("bge_vector", {}).get("hits", [])
            sage_hits = m_evac.get("graphsage_hybrid", {}).get("hits", [])
            gps_hits = m_evac.get("graph_gps_hybrid", {}).get("hits", [])
            
            bge_qs = {h['question'] for h in bge_hits}
            sage_overlap = len({h['question'] for h in sage_hits}.intersection(bge_qs))
            gps_overlap = len({h['question'] for h in gps_hits}.intersection(bge_qs))
            
            st.markdown(f"<div style='color: #94a3b8;'>Shared with BGE → GraphSAGE: {sage_overlap} | GraphGPS: {gps_overlap}</div>", unsafe_allow_html=True)

            st.write("---")
            st.subheader("GT preview (LLM-style with reflection)")
            
            gt_examples = payload.get("gt_examples", [])
            if gt_examples:
                for gt in gt_examples[:2]:
                    st.markdown(f"""
                    <div style="background-color: #1e2530; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #2b3342;">
                        <div style="color: #38bdf8; font-weight: 600; margin-bottom: 5px;">Q:</div>
                        <div style="color: #e2e8f0; margin-bottom: 10px;">{gt['question']}</div>
                        <div style="color: #38bdf8; font-weight: 600; margin-bottom: 5px;">GT:</div>
                        <div style="color: #e2e8f0; margin-bottom: 5px;">{gt['gt_answer']}</div>
                        <div style="color: #38bdf8; font-weight: 600; margin-bottom: 5px;">Reflection:</div>
                        <div style="color: #94a3b8;">{gt['reflection']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with st.expander("See Node Data"):
                    st.dataframe(payload["graph_sample"]["nodes"])
                    
    else:
        # Empty State
        st.markdown(
            """
            <div style="text-align: center; margin-top: 50px; color: #64748b;">
                <h3>Ready to Run</h3>
                <p>Click <b>Quick Run</b> in the sidebar to launch the experimental pipeline.</p>
            </div>
            """, unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
