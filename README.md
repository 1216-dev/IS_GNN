# Persona-aware Evacuation Explorer

A comparative research platform exploring the effectiveness of Graph Neural Networks (GNNs) versus Vector RAG for complex, persona-driven evacuation scenarios.

## 🚀 Overview

This application demonstrates a "Persona-aware" retrieval system that bakes demographic signals (Income, Race, Education) directly into the graph structure. It compares three retrieval methodologies:

1.  **BGE Vector (Baseline)**: Standard dense vector retrieval using BGE-small embeddings.
2.  **GraphSAGE Hybrid**: GNN-based re-ranking leveraging node neighborhoods.
3.  **Transformer-GNN Hybrid (GraphGPS)**: Advanced graph transformer model capturing global graph structures and positional encodings.

## ✨ Features

-   **Interactive Dashboard**: Built with Streamlit for real-time visualization.
-   **Heterogeneous Graph Construction**: Models relationships between Personas, Demographics, Incidents, and Facilities.
-   **Persona Simulation**: Generates synthetic queries conditioned on specific persona constraints (e.g., "Equity Analyst" analyzing "Low Income" neighborhoods).
-   **Overlap Analysis**: Visualizes the intersection of retrieved documents across different models.
-   **Ground Truth Reflection**: Displays LLM-generated ground truth with reasoning trace.

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/1216-dev/IS_GNN.git
    cd IS_GNN
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

    *Note: Ensure you have `streamlit` installed if not covered by requirements.*

## 🏃 Usage

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

Open your browser at `http://localhost:8501`.

## 📂 Project Structure

-   `streamlit_app.py`: Main application entry point and UI logic.
-   `pipeline.py`: Orchestrator for data loading, model inference, and evaluation.
-   `gnn_model.py`: PyTorch Geometric implementations of GraphSAGE and GraphGPS.
-   `graph_builder.py`: Logic for constructing the heterogeneous knowledge graph.
-   `personas.py`: Definitions and logic for synthetic persona generation.
-   `rag_eval.py`: Evaluation metrics (Recall@K, MRR) and ground truth generation.

## 🔬 Research Mode

Toggle "Research Mode" in the sidebar to enable experimental features like Semantic Edge weighting and Z-Score fusion for hybrid retrieval.
