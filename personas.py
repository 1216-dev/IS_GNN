"""Persona definitions and synthetic Q&A generation."""

from __future__ import annotations


import random
import json

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class Persona:
    name: str
    domain: str
    interests: List[str]
    focus_metrics: List[str]


def base_personas() -> List[Persona]:
    return [
        Persona(
            name="Emergency Manager",
            domain="response",
            interests=["incident hotspots", "evacuation routes", "shelter surge", "multi-agency coord"],
            focus_metrics=["response_time", "incident_rate", "shelter_capacity"],
        ),
        Persona(
            name="City Planner",
            domain="planning",
            interests=["land use", "population density", "transport chokepoints", "critical facilities"],
            focus_metrics=["density", "vehicle_access", "facility_distance"],
        ),
        Persona(
            name="Public Health Officer",
            domain="health",
            interests=["elderly", "disability", "language access", "clinic proximity"],
            focus_metrics=["age_65_plus", "disability_rate", "language_isolation", "clinic_distance"],
        ),
        Persona(
            name="Social Worker",
            domain="social",
            interests=["low income", "rent burden", "food access", "homeless services"],
            focus_metrics=["income_lt_30k", "rent_burden", "food_access", "shelter_access", "race_equity_index"],
        ),
        Persona(
            name="Transit Ops",
            domain="mobility",
            interests=["vehicle ownership", "subway coverage", "bus redundancy", "evac bus staging"],
            focus_metrics=["vehicle_access", "transit_coverage", "evac_route_redundancy"],
        ),
        Persona(
            name="Utility Ops",
            domain="utilities",
            interests=["power resilience", "water main breaks", "fuel supply", "backup gen coverage"],
            focus_metrics=["grid_redundancy", "water_break_rate", "fuel_sites", "generator_sites"],
        ),
        Persona(
            name="Equity Analyst",
            domain="equity",
            interests=["income gaps", "racial disparities", "education access"],
            focus_metrics=["income_decile", "cvap_race_mix", "education_attainment"],
        ),
    ]



def generate_questions(persona: Persona, n: int = 120) -> List[str]:
    """Load complex synthetic questions from the pre-generated JSON file."""
    try:
        with open("data/synthetic_persona_data.json", "r") as f:
            all_data = json.load(f)
    except FileNotFoundError:
        # Fallback if file missing (e.g. initial run)
        return []

    # Filter by persona name
    persona_qs = [item["question"] for item in all_data if item["persona"] == persona.name]
    
    # If we need more than available, we cycle; if less, we sample
    if not persona_qs:
        return []
    
    if n > len(persona_qs):
        # Repeat to fill
        results = (persona_qs * (n // len(persona_qs) + 1))[:n]
    else:
        results = random.sample(persona_qs, n)
    return results


def synthesize_persona_examples(
    persona: Persona,
    demo: pd.DataFrame,
    incidents: pd.DataFrame,
    facilities: pd.DataFrame,
    n: int = 120,
    borough_stats: Dict[str, Dict[str, str]] | None = None,
) -> List[Dict]:
    """Generate synthetic QA pairs using pre-generated complex questions."""
    qs = generate_questions(persona, n=n)
    examples: List[Dict] = []
    
    # If no questions found (e.g. persona name mismatch or file missing), return empty
    if not qs:
        return []

    for q in qs:
        fac = facilities.sample(1).iloc[0]
        inc = incidents.sample(1).iloc[0]
        boro = fac.get("boro", "unknown")
        stats = borough_stats.get(boro, {}) if borough_stats else {}
        income_tag = stats.get("income_tag", "median")
        race_tag = stats.get("race_tag", "mixed")
        edu_tag = stats.get("education_tag", "edu_score:0.5")
        cap = fac.get("cap", "n/a")
        facname = fac.get("facname", "unknown")
        inc_type = inc.get("incident_type", inc.get("type", "incident"))
        inc_boro = inc.get("borough", inc.get("boro", "unknown"))
        risk = random.choice(["high", "moderate", "rising"])
        
        # We construct a synthetic answer that tries to "answer" the complex question 
        # by hallucinating specific details relevant to the question's complexity.
        # ideally this would be an LLM, but for this task we use a structured template 
        # that mimics a thoughtful response.
        
        ans = (
            f"Analysis for {persona.name}: The critical node is {facname} in {boro} (capacity:{cap}). "
            f"Under {risk} situational risk, the {inc_type} pattern in {inc_boro} exacerbates vulnerabilities. "
            f"Contextual factors: {income_tag} income levels, {race_tag} demographics, and {edu_tag} typically "
            f"correlate with reduced resilience in these scenarios. "
            f"Recommendation: Prioritize resource allocation to overlap with {inc_type} zones."
        )
        
        examples.append(
            {
                "persona": persona.name,
                "question": q,
                "answer": ans,
                "facility_idx": int(getattr(fac, "name", fac.get("id", 0))),
                "incident_idx": int(getattr(inc, "name", inc.get("id", 0))),
                "income_tag": income_tag,
                "race_tag": race_tag,
                "education_tag": edu_tag,
            }
        )
    return examples

