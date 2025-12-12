
import json
import random

personas = [
    {
        "name": "Emergency Manager",
        "domain": "response",
        "interests": ["incident hotspots", "evacuation routes", "shelter surge", "multi-agency coord"],
        "style": "urgent, tactical, safety-first"
    },
    {
        "name": "City Planner",
        "domain": "planning",
        "interests": ["land use", "population density", "transport chokepoints", "critical facilities"],
        "style": "strategic, long-term, spatial"
    },
    {
        "name": "Public Health Officer",
        "domain": "health",
        "interests": ["elderly", "disability", "language access", "clinic proximity"],
        "style": "human-centric, vulnerable-population-focused"
    },
    {
        "name": "Social Worker",
        "domain": "social",
        "interests": ["low income", "rent burden", "food access", "homeless services"],
        "style": "advocacy, equity-focused, resource-oriented"
    },
    {
        "name": "Transit Ops",
        "domain": "mobility",
        "interests": ["vehicle ownership", "subway coverage", "bus redundancy", "evac bus staging"],
        "style": "logistical, network-oriented, efficiency-focused"
    },
    {
        "name": "Utility Ops",
        "domain": "utilities",
        "interests": ["power resilience", "water main breaks", "fuel supply", "backup gen coverage"],
        "style": "technical, infrastructure-heavy, reliability-focused"
    },
     {
        "name": "Equity Analyst",
        "domain": "equity",
        "interests": ["income gaps", "racial disparities", "education access", "historical underinvestment"],
        "style": "analytical, justice-oriented, systemic"
    }
]

# "Convoluted" templates
templates = [
    "Given the intersection of {interest} and high {metric}, how does the lack of {interest2} in {context_tag} neighborhoods exacerbate the failure of {facility}?",
    "If we consider the cascading effects of a failure in {facility} on {interest}, particularly in areas defined by {metric} and {context_tag}, what are the secondary risks to {interest2}?",
    "Analyze the spatial correlation between {interest} hotspots and {metric} to determine if {facility} is adequately positioned to serve {context_tag} populations during a {interest2} crisis.",
    "Despite the apparent resilience of {facility}, how does the underlying {metric} in the surrounding {context_tag} community undermine effective {interest} when faced with {interest2}?",
    "When evaluating {interest}, why does the metric of {metric} fail to capture the true vulnerability of {context_tag} residents reliant on {facility} for {interest2}?",
    "Considering the {context_tag} demographic's reliance on {interest}, how functionality of {facility} is critical when {metric} indicates a shortage of {interest2}?",
    "In a scenario where {interest} is compromised, how does the {metric} of the local population interact with the capacity of {facility} to create a bottleneck for {interest2}?",
    "Reflecting on the historical underperformance of {interest} systems, what does the {metric} data suggest about the equitable distribution of {facility} access for {context_tag} groups needing {interest2}?",
    "How does the {metric}-driven logic of current {interest} planning overlook the nuances of {context_tag} communities who depend on {facility} for {interest2} stability?",
    "Map the dependency chain from {facility} failure to {interest} collapse, specifically highlighting how {metric} intensities in {context_tag} zones accelerate the loss of {interest2}."
]

metrics = ["population density", "incident frequency", "response time lag", "poverty rate", "vehicle unavailability", "elderly concentration", "flood risk score", "grid instability", "transit desert index"]
context_tags = ["low-income", "minority-majority", "elderly-dense", "transit-dependent", "linguistically isolated", "historically redlined", "high-density", "industrial-adjacent"]

data = []

for p in personas:
    for _ in range(110): # Generate ~110 questions per persona
        interest = random.choice(p["interests"])
        interest2 = random.choice([i for i in p["interests"] if i != interest] + ["general resilience"])
        metric = random.choice(metrics)
        facility = "the local hospital" # Placeholder, will be replaced dynamically or kept generic
        context = random.choice(context_tags)
        
        tmpl = random.choice(templates)
        q = tmpl.format(
            interest=interest,
            interest2=interest2,
            metric=metric,
            facility=facility,
            context_tag=context
        )
        
        data.append({
            "persona": p["name"],
            "domain": p["domain"],
            "question": q,
            "interests_used": [interest, interest2],
            "metric_used": metric,
            "context_tag_used": context
        })

with open("data/synthetic_persona_data.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Generated {len(data)} synthetic questions.")
