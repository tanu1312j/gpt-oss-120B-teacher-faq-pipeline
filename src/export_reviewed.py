#!/usr/bin/env python3
"""Export only manually approved teacher answers into a future SFT chat JSONL.

No training code is included or run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def export(source: Path, approved_ids: Path, target: Path) -> int:
    if not source.exists() or not approved_ids.exists():
        raise ValueError("Both source JSONL and approved ID file must exist")
    approved = {
        line.strip() for line in approved_ids.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not approved:
        raise ValueError("No IDs approved; inspect teacher_outputs.jsonl first")
    records: dict[str, dict] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["id"] in records:
            raise ValueError(f"Duplicate teacher output ID {rec['id']}")
        records[rec["id"]] = rec
    missing_ids = approved - records.keys()
    if missing_ids:
        raise ValueError(f"Approved IDs not found: {sorted(missing_ids)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for rec_id, rec in records.items():
            if rec_id not in approved:
                continue
            if (not rec.get("teacher_answer") or
                rec["teacher_answer"] == "INSUFFICIENT_SOURCE_INFORMATION" or
                rec.get("finish_reason") == "length"):
                raise ValueError(f"Do not export an empty/insufficient/truncated answer: {rec_id}")
            conversation = {"messages": [
                {"role": "user", "content": rec["question"]},
                {"role": "assistant", "content": rec["teacher_answer"]},
            ]}
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
    return len(approved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/teacher_outputs.jsonl"))
    parser.add_argument("--approved-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/approved_sft.jsonl"))
    args = parser.parse_args()
    number = export(args.input, args.approved_ids, args.output)
    print(f"Exported {number} manually approved samples to {args.output}")


if __name__ == "__main__":
    main()
