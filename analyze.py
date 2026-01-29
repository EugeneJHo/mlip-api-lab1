import json
import os
from typing import Any, Dict
from litellm import completion

MODEL = "groq/llama-3.3-70b-versatile"

def get_itinerary(destination: str) -> Dict[str, Any]:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Missing GROQ_API_KEY. Put it in your .env file.")

    prompt = f"""
Return ONLY valid JSON (no markdown, no backticks) for a travel summary of {destination}.
Must match this schema exactly:

{{
  "destination": "{destination}",
  "price_range": "budget|mid-range|luxury",
  "ideal_visit_times": ["..."],
  "top_attractions": ["..."]
}}

Rules:
- ideal_visit_times: 2–4 items
- top_attractions: 5–8 items
"""

    resp = completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
    )

    content = resp["choices"][0]["message"]["content"].strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model returned invalid JSON: {e}. Raw output: {content}")

    required = {"destination", "price_range", "ideal_visit_times", "top_attractions"}
    if not required.issubset(data):
        raise RuntimeError(f"Missing required keys. Got: {list(data.keys())}")

    return data
