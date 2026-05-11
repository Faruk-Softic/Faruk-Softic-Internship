"""
This script just generates the pipelines.json file
"""

import json
import itertools
import os

# If needed, change paths here
OUTPUT_FILE = r"C:\Users\fsoft\Desktop\Za Internship - code\pipelines.json"
# ─────────────────────────────────────────────────────────────────────────────

# Define default temp. for models that don't support it
NO_TEMP_MODELS  = ("gpt-5",)        # startswith check — covers gpt-5, gpt-5-nano, etc.
DEFAULT_TEMPERATURE = 1.0

# Define parameters to be used for each test

parameters = {
    "model":          ["gpt-5.4-nano"],   
    "temperature":    [1.0],             # single value for now
    "rubric_version": ["original", "improved"],
    "grading_mode":   ["sequential", "holistic"],
}

# Make all combinations
keys   = list(parameters.keys())
values = list(parameters.values())

pipelines = []
for combo in itertools.product(*values):
    pipeline = dict(zip(keys, combo))
    model       = pipeline["model"]
    temperature = pipeline["temperature"]

    # Skip temperature variation for models that don't support it
    if any(model.startswith(m) for m in NO_TEMP_MODELS) and temperature != DEFAULT_TEMPERATURE:
        continue

    # Build a human-readable pipeline ID
    pipeline_id = (
        f"{model}"
        f"_t{temperature}"
        f"_{pipeline['rubric_version']}"
        f"_{pipeline['grading_mode']}"
    )
    pipeline["pipeline_id"] = pipeline_id
    pipelines.append(pipeline)

print(f"Generated {len(pipelines)} pipelines:")
for p in pipelines:
    print(f"  {p['pipeline_id']}")

# Save to json
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(pipelines, f, indent=2)

print(f"\nSaved to: {OUTPUT_FILE}")
