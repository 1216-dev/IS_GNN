"""Graph construction for neighborhoods, facilities, and incidents."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data


def build_graph(
    demo: pd.DataFrame, incidents: pd.DataFrame, facilities: pd.DataFrame
) -> Tuple[Data, Dict[str, int], List[str], Dict[int, int], Dict[int, int], torch.Tensor, List[int]]:
    """
    Build a simple homogeneous graph:
      - nodes: neighborhoods (from demographics), facilities, incidents
      - edges: facility->neighborhood, incident->neighborhood (undirected for simplicity)
    Feature assembly is simplified; extend with engineered features as needed.
    """
    neighborhoods = demo["geographic_area_-_borough"].fillna("unknown").astype(str).unique()
    nid_map = {n: i for i, n in enumerate(neighborhoods)}
    node_feats: List[List[float]] = []
    node_types: List[str] = []

    # Neighborhood features: population change percent as single feature for now
    demo_group = demo.groupby("geographic_area_-_borough")["total_population_change_2000-2010_percent"].mean()
    income_cols = [c for c in demo.columns if "income" in c]
    education_cols = [c for c in demo.columns if "education" in c or "bachelor" in c]
    race_cols = [c for c in demo.columns if "race" in c or "ethnic" in c]
    income_group = demo.groupby("geographic_area_-_borough")[income_cols].mean() if income_cols else None
    edu_group = demo.groupby("geographic_area_-_borough")[education_cols].mean() if education_cols else None
    race_group = demo.groupby("geographic_area_-_borough")[race_cols].mean() if race_cols else None
    for n in neighborhoods:
        income_val = float(income_group.loc[n].mean()) if income_group is not None and n in income_group.index else 0.0
        edu_val = float(edu_group.loc[n].mean()) if edu_group is not None and n in edu_group.index else 0.0
        race_val = float(race_group.loc[n].mean()) if race_group is not None and n in race_group.index else 0.0
        node_feats.append([demo_group.get(n, 0.0), income_val, edu_val, race_val])
        node_types.append("neighborhood")

    edges_src: List[int] = []
    edges_dst: List[int] = []
    edge_types: List[int] = []
    cur_idx = len(node_feats)

    # Facilities nodes and edges
    facility_map: Dict[int, int] = {}
    for row_idx, row in facilities.iterrows():
        capacity = float(row.get("cap", 0.0)) if "cap" in row else 0.0
        node_feats.append([capacity, 0.0, 0.0, 0.0])
        node_types.append("facility")
        fac_idx = cur_idx
        facility_map[row_idx] = fac_idx
        cur_idx += 1
        neigh = row.get("boro", "unknown")
        neigh_idx = nid_map.get(neigh, 0)
        edges_src.extend([fac_idx, neigh_idx])
        edges_dst.extend([neigh_idx, fac_idx])
        edge_types.extend([0, 1])

    # Incident nodes and edges
    incident_map: Dict[int, int] = {}
    for row_idx, row in incidents.iterrows():
        severity = 1.0  # placeholder
        node_feats.append([severity, 0.0, 0.0, 0.0])
        node_types.append("incident")
        inc_idx = cur_idx
        incident_map[row_idx] = inc_idx
        cur_idx += 1
        neigh = row.get("borough", "unknown") if "borough" in row else row.get("boro", "unknown")
        neigh_idx = nid_map.get(neigh, 0)
        edges_src.extend([inc_idx, neigh_idx])
        edges_dst.extend([neigh_idx, inc_idx])
        edge_types.extend([2, 3])


    # Spatial edges between facilities (naive O(N^2) but acceptable for small N < 2000)
    # If N is large, we should use a KDTree or limit this. 
    # Let's limit to top K nearest neighbors or distance threshold.
    
    fac_coords = []
    ids = []
    for row_idx, row in facilities.iterrows():
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            if not pd.isna(lat) and not pd.isna(lon):
                fac_coords.append((lat, lon))
                ids.append(facility_map[row_idx])
        except (ValueError, KeyError):
            continue


    import numpy as np

    # Optimization: Only consider top 800 separate facilities for spatial edges to keep it fast
    # or use simple bucketing if we want coverage. 
    # For now: downsample fac_coords to random 800 or top capacity
    
    if len(fac_coords) > 800:
        # Just take a subset to avoid N^2 where N=34k
        import random
        indices = random.sample(range(len(fac_coords)), 800)
        fac_coords_subset = [fac_coords[i] for i in indices]
        ids_subset = [ids[i] for i in indices]
    else:
        fac_coords_subset = fac_coords
        ids_subset = ids

    if len(fac_coords_subset) > 1:
        coords = torch.tensor(fac_coords_subset)
        # Simple Euclidean on lat/lon
        dist_threshold = 0.01  
        
        # Calculate pairwise distance for subset
        for i in range(len(coords)):
            # Vectorized distance to all subsequent nodes is faster than nested loop
            if i + 1 >= len(coords):
                break
            dists = torch.norm(coords[i+1:] - coords[i], dim=1)
            mask = dists < dist_threshold
            
            # Get indices where mask is True
            matches = torch.nonzero(mask).squeeze(1) + (i + 1)
            
            if len(matches) > 0:
                src_node = ids_subset[i]
                dst_nodes = [ids_subset[m] for m in matches]
                
                # Add edges
                edges_src.extend([src_node] * len(dst_nodes))
                edges_dst.extend(dst_nodes)
                edges_src.extend(dst_nodes)
                edges_dst.extend([src_node] * len(dst_nodes))
                edge_types.extend([4] * (2 * len(dst_nodes)))



    # Semantic/Domain edges (heuristic)
    # Connect incidents to facilities matching domain keywords
    # This is a "Research-Oriented" feature for high accuracy
    
    # Simple keyword map
    domain_keywords = {
        "HEALTH": ["Health", "Hospital", "Medical"],
        "SAFETY": ["Police", "Fire", "Safety", "Structural"],
        "UTILITY": ["Water", "Electric", "Utility", "Power"],
        "TRANSIT": ["Transit", "Bus", "Subway", "Rail"],
        "COMMUNITY": ["Park", "Community", "Recreation", "School"]
    }

    # Pre-index facilities by domain
    fac_indices_by_domain = {k: [] for k in domain_keywords}
    for row_idx, row in facilities.iterrows():
        fgrp = str(row.get("facgroup", "")).upper()
        fsub = str(row.get("facsubgrp", "")).upper()
        ftype = str(row.get("factype", "")).upper()
        
        for dom, keys in domain_keywords.items():
            if any(k.upper() in fgrp or k.upper() in fsub or k.upper() in ftype for k in keys):
                fac_indices_by_domain[dom].append(facility_map[row_idx])

    # Connect incidents to these facilities
    # Only connect to a sample (3) to avoid explosion
    import random
    
    for row_idx, row in incidents.iterrows():
        itype = str(row.get("Incident Type", "")).upper()
        inc_node_idx = incident_map[row_idx]
        
        for dom, keys in domain_keywords.items():
            if any(k.upper() in itype for k in keys):
                candidates = fac_indices_by_domain[dom]
                if candidates:
                    # Pick 3 random matching facilities (e.g. incident 'Water Main' -> 3 water facilities)
                    # This tells the GNN "this incident interacts with this TYPE of infrastructure"
                    chosen = random.sample(candidates, min(len(candidates), 3))
                    
                    edges_src.extend([inc_node_idx] * len(chosen))
                    edges_dst.extend(chosen)
                    edges_src.extend(chosen)
                    edges_dst.extend([inc_node_idx] * len(chosen))
                    edge_types.extend([5] * (2 * len(chosen))) # Type 5: Semantic

    x = torch.tensor(node_feats, dtype=torch.float)
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_types, dtype=torch.long).view(-1, 1)

    # Simple positional encodings based on node ids (sin/cos).
    node_ids = torch.arange(x.shape[0], dtype=torch.float).view(-1, 1)
    pe = torch.cat([torch.sin(node_ids / 10.0), torch.cos(node_ids / 10.0)], dim=1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data, nid_map, node_types, facility_map, incident_map, pe, edge_types


