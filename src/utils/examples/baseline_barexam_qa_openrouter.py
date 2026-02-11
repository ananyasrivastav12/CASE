"""
Simple script to query via OpenRouter
using the reglab/barexam_qa dataset — no retrieval or embeddings.
"""

import os
import json
import time
import requests
from datasets import load_from_disk, load_dataset
from tqdm import tqdm

# ============================================================
# 1. Configuration
# ============================================================

# Set your API key (get it from https://openrouter.ai/keys)
os.environ["OPENROUTER_API_KEY"] = "enter your key"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL_NAME = "openai/gpt-oss-20b:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Output file
OUTPUT_FILE = "barexam_qa_no_rag.jsonl"

# Number of questions to test (set None for full dataset)
NUM_EXAMPLES = 10

# ============================================================
# 2. Load dataset
# ============================================================

print("Loading dataset...")
# qa_dataset = load_dataset("reglab/barexam_qa", name="qa", split="train")
qa_dataset = load_from_disk("barexam_qa_qa")

qa_dataset = qa_dataset["train"]

if NUM_EXAMPLES:
    qa_dataset = qa_dataset.select(range(NUM_EXAMPLES))

print(f"Loaded {len(qa_dataset)} questions.\n")

# ============================================================
# 3. Define the query function
# ============================================================

def query(question):
    """Query via OpenRouter."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # Construct the conversation payload
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an expert legal assistant answering bar exam questions clearly and accurately."},
            {"role": "user", "content": question},
        ],
        "temperature": 0.3,
        "max_tokens": 400,
    }

    try:
        print(json.dumps(payload, indent=2))
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"

# ============================================================
# 4. Loop over dataset and query
# ============================================================

results = []

print("Querying...")

for i, row in enumerate(tqdm(qa_dataset, desc="Questions")):
    question = row.get("question", "")
    gold_answer = row.get("answer", "")

    model_response = query(question)

    result = {
        "id": i,
        "question": question,
        "gold_answer": gold_answer,
        "model_response": model_response,
    }

    results.append(result)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    time.sleep(5.0)  # gentle rate limit

print(f"\nDone! Saved {len(results)} responses to {OUTPUT_FILE}")
