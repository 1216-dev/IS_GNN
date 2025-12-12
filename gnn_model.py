"""Graph baselines: GraphSAGE (2017) and a modern GPS (GraphGPS/Transformer-style) GNN."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv, TransformerConv, GPSConv, GINEConv
from torch_geometric.data import Data


class SimpleSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.act(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x



def train_gnn(data: Data, epochs: int = 20, lr: float = 1e-3, out_dim: int = 64) -> torch.Tensor:
    """
    Trains GraphSAGE using Contrastive Learning (InfoNCE).
    This pushes connected nodes (positives) closer and random nodes (negatives) apart.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = SimpleSAGE(in_dim=data.num_node_features, out_dim=out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Simple InfoNCE implementation
    # We treat edges as positive pairs. 
    # For every node, its neighbors are positives, other nodes are negatives.
    


    # Re-enabling InfoNCE (Contrastive Learning) per user research report
    # This pushes connected nodes (positives) closer and random nodes (negatives) apart.
    
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        
        # Sample edges for positive pairs
        perm = torch.randperm(data.edge_index.size(1))
        edges = data.edge_index[:, perm[:2000]] # Batch of 2000 edges
        
        src_emb = out[edges[0]]
        dst_emb = out[edges[1]]
        
        # Cosine similarity for positives
        pos_sim = torch.nn.functional.cosine_similarity(src_emb, dst_emb, dim=-1)
        
        # Negatives: shift dst randomly
        neg_dst_idx = torch.randint(0, out.size(0), (edges.size(1),), device=device)
        neg_dst_emb = out[neg_dst_idx]
        neg_sim = torch.nn.functional.cosine_similarity(src_emb, neg_dst_emb, dim=-1)
        
        # InfoNCE Loss: -log(sigmoid(pos)) - log(sigmoid(-neg))
        loss = -torch.log(torch.sigmoid(pos_sim) + 1e-15).mean() - torch.log(torch.sigmoid(-neg_sim) + 1e-15).mean()
        
        loss.backward()
        opt.step()
        
    embeddings = model(data.x, data.edge_index).detach().cpu()
    return embeddings





class GraphTransformer(nn.Module):
    """Lightweight transformer-style graph encoder using TransformerConv."""

    def __init__(self, in_dim: int, hidden: int = 96, out_dim: int = 96, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = max(hidden, heads)  # ensure divisibility
        self.conv1 = TransformerConv(in_dim, hidden // heads, heads=heads, dropout=dropout, beta=True)
        self.conv2 = TransformerConv(hidden, out_dim // heads, heads=heads, dropout=dropout, beta=True)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.conv2(x, edge_index)
        x = self.norm(x)
        return x



def train_graph_transformer(data: Data, epochs: int = 20, lr: float = 1e-3, out_dim: int = 96) -> torch.Tensor:
    """
    Train a transformer-based GNN using Contrastive Learning (InfoNCE).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = GraphTransformer(in_dim=data.num_node_features, out_dim=out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    


    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        
        # Contrastive Logic
        perm = torch.randperm(data.edge_index.size(1))
        edges = data.edge_index[:, perm[:2000]] 
        src = out[edges[0]]
        dst = out[edges[1]]
        neg = out[torch.randint(0, out.size(0), (edges.size(1),), device=device)]
        
        pos_score = (src * dst).sum(dim=-1)
        neg_score = (src * neg).sum(dim=-1)
        
        # Margin ranking loss: max(0, 1 - pos + neg)
        loss = torch.clamp(1.0 - pos_score + neg_score, min=0).mean()
        
        loss.backward()
        opt.step()
        
    embeddings = model(data.x, data.edge_index).detach().cpu()
    return embeddings





class GPSGraph(nn.Module):
    """GraphGPS-style encoder using GPSConv (Transformer + message passing)."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 96,
        out_dim: int = 96,
        heads: int = 4,
        dropout: float = 0.1,
        num_edge_types: int = 4,
        pos_dim: int = 0,
    ):
        super().__init__()
        embed_dim = hidden
        self.input = nn.Linear(in_dim + pos_dim, embed_dim)
        self.edge_emb = nn.Embedding(num_edge_types, embed_dim)
        mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.conv1 = GPSConv(
            channels=embed_dim,
            conv=GINEConv(nn=mlp),
            heads=heads,
            dropout=dropout,
        )
        self.conv2 = GPSConv(
            channels=embed_dim,
            conv=GINEConv(nn=mlp),
            heads=heads,
            dropout=dropout,
        )
        self.out = nn.Linear(embed_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.pos_dim = pos_dim

    def forward(self, x, edge_index, edge_attr=None, pos_enc=None):
        if pos_enc is not None:
            x = torch.cat([x, pos_enc.to(x.device)], dim=1)
        x = self.input(x)
        eattr = None
        if edge_attr is not None:
            eattr = self.edge_emb(edge_attr.squeeze().long())
        x = self.conv1(x, edge_index, edge_attr=eattr)
        x = self.conv2(x, edge_index, edge_attr=eattr)
        x = self.out(x)
        x = self.norm(x)
        return x


def train_graph_gps(
    data: Data,
    epochs: int = 8,
    lr: float = 1e-3,
    out_dim: int = 96,
    edge_attr: torch.Tensor | None = None,
    pos_enc: torch.Tensor | None = None,
    research_mode: bool = False,
) -> torch.Tensor:
    """Train a GraphGPS (attention + message passing) encoder as the modern baseline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    pos_dim = pos_enc.shape[1] if (research_mode and pos_enc is not None) else 0
    model = GPSGraph(
        in_dim=data.num_node_features,
        out_dim=out_dim,
        num_edge_types=int(edge_attr.max().item()) + 1 if edge_attr is not None else 4,
        pos_dim=pos_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.MSELoss()
    edge_attr_dev = edge_attr.to(device) if edge_attr is not None else None
    pos_dev = pos_enc.to(device) if (pos_enc is not None and research_mode) else None




    # GPS Contrastive Logic
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index, edge_attr=edge_attr_dev, pos_enc=pos_dev)
        
        perm = torch.randperm(data.edge_index.size(1))
        edges = data.edge_index[:, perm[:2500]] 
        src = out[edges[0]]
        dst = out[edges[1]]
        neg = out[torch.randint(0, out.size(0), (edges.size(1),), device=device)]
        
        pos_score = torch.nn.functional.cosine_similarity(src, dst, dim=-1)
        neg_score = torch.nn.functional.cosine_similarity(src, neg, dim=-1)
        
        tau = 0.5
        logits = torch.cat([pos_score.unsqueeze(1), neg_score.unsqueeze(1)], dim=1) / tau
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        
        loss.backward()
        opt.step()
    embeddings = model(data.x, data.edge_index, edge_attr=edge_attr_dev, pos_enc=pos_dev).detach().cpu()
    return embeddings



