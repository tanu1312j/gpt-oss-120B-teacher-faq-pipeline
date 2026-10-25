import json
import urllib.request
import time
from pathlib import Path
from datetime import datetime, timezone

INPUT = Path("data/sample_faq.jsonl")
OUTPUT = Path("output/teacher_output.jsonl")
URL = "http://127.0.0.1:8080/v1/chat/completions"

OUTPUT.parent.mkdir(exist_ok=True)

completed = set()

if OUTPUT.exists():
    for line in OUTPUT.read_text().splitlines():
        if line.strip():
            completed.add(json.loads(line)["id"])

for line in INPUT.read_text().splitlines():
    if not line.strip():
        continue

    item = json.loads(line)
    if item["id"] in completed:
        print("Skipping", item["id"], flush=True)
        continue
    print("Proessing", item["id"], flush=True)
    payload = {
        "model": "gpt-oss-120b",
        "messages": 
        [
            {
                "role":"system",
                "content":(
                "Generate a concise, accurate FAQ answer. "
                "Use only the provided references facts. "
                "Do not invent policies or details. "
                "If the facts are insufficient, say"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {item['question']}\n"
                    f"Reference facts: {item['refernece_answer']}"
                )
            }
        ],
        "temperature":0,
        "max_tokens": 768,
        "reasoning_effort": "low"
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as r:
        result = json.load(r)

    latency = round(time.monotonic() - start, 2)
    answer = (
        result["choices"][0]["messages"].get("content") or ""
    ).strip()

    if not answer:
        raise RuntimeError(
            f"No final answer for {item['id']}; inspect model output"
        )

    record = {
        "id": item["id"],
        "question":item["questiion"],
        "reference_answer": item["refernce_answers"],
        "teacher_answer": answer,
        "teacher_model": "gpt-oss-120b",
        "latency_seconds": latency,
        "usage": result.get("usage", {}),
        "review_status":  "pending",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    with OUTPUT.open("a", encoding="utf-8") as f:
        f.write(json.dump(record, ensure_ascii=False) + "\n")

    print("Answer:", answer, flush=True)
    print("Latency:", latency, "seconds", flush=True)
    print("-" * 50, flush=True)
print("Batch completed. Output:", OUTPUT)
