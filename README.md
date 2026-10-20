# GPT-OSS-120B teacher FAQ dataset generator (AWS Graviton, offline batch)

**Phase 1 only.** Generate source-grounded candidate teacher answers with `gpt-oss-120b`, review them, and export approved examples for possible later student training. **This repository does not train or run GPT-OSS-20B.**

## Architecture

```text
Synthetic sample FAQ CSV (questions + reference facts)
          |
          v
Amazon EC2 Graviton4 r8g.4xlarge (ARM64 CPU, 128 GiB RAM)
          |
          v
llama.cpp + GPT-OSS-120B MXFP4 GGUF (~63.4 GB download)
          |   localhost:8080, transient batch-only process
          v
src/generate_teacher.py -- sequential inference, checkpoint and resume
          |
          v
output/teacher_outputs.jsonl (answers + source + flags, awaiting human review)
          |
          v
Manual approval (output/approved_ids.txt)
          |
          v
src/export_reviewed.py -> output/approved_sft.jsonl
```

## Before you start

- Use a Graviton **ARM64** instance such as EC2 `r8g.4xlarge` (Graviton4, 16 vCPU, 128 GiB RAM); if CPU performance is insufficient, benchmark or switch to a GPU-backed teacher job. Smaller Graviton instances may not fit the model.
- Amazon Linux 2023 ARM64 AMI and an EBS volume with adequate free space (recommend **150 GiB or more** for checkpoint download, builds, and artifacts). Provision adequate storage throughput.
- Inbound security group: SSH from your trusted IP or use SSM; **do not open 8080**. `llama-server` binds localhost only.
- CPU inference for a 120B model can be slow. Benchmark 1-2 questions before processing a large corpus.
- This is **not air-gapped installation**: Git, dependencies and model are downloaded once; *inference itself* is local without a remote LLM API.
- Sample FAQ is **fictional**. Replace it with licensed and authoritative answers, not private customer data unless your security approval permits this.

## 1 — Copy repository to your EC2 and set up

```bash
# When repository is published, on your EC2 instance:
git clone https://github.com/<YOUR_GITHUB_USERNAME>/gpt-oss-teacher-faq.git
cd gpt-oss-teacher-faq
bash scripts/setup_graviton.sh
```

This compiles `llama.cpp` for your Graviton processor with `-mcpu=native` and installs the model-download CLI in `.venv`. Use your own clone URL.

## 2 — Download the teacher checkpoint once

```bash
bash scripts/download_model.sh
```

Repository: `ggml-org/gpt-oss-120b-GGUF`; file: `gpt-oss-120b-MXFP4.gguf`. Large files are **never pushed to Git**. Confirm available storage before downloading.

## 3 — Edit your input FAQs

Update `data/sample_faq.csv` or supply your own CSV with `id`, `question`, `reference_answer`, and optional `category`. Each row must provide trusted reference facts. The teacher will write a concise answer *based on those facts* rather than guess policy from its model weights.

## 4 — Offline/batch generation (no public serving)

```bash
# Smoke-test only the first two rows:
bash scripts/run_batch.sh --limit 2

# Then resume the remaining FAQ rows. No need to redo those two.
bash scripts/run_batch.sh

# For a different CSV:
bash scripts/run_batch.sh --input data/my_faq.csv --output output/my_teacher_outputs.jsonl
```

The script starts `llama-server` on `127.0.0.1:8080`, loads weights once, sequentially calls it, checkpoints every successful answer, and stops the server even when the batch ends/errors. Localhost HTTP is an internal process interface, **not a deployed live/public endpoint**. Model reasoning, if present in `reasoning_content`, is intentionally excluded from saved data.

If a batch is interrupted, run the same command to resume. If a FAQ question/reference fact changes, choose a **new output filename** to avoid silently mixing versions.

Check `logs/llama-server.log` for start/inference problems.

## 5 — Review candidate answers and export for later training

Every response is marked `pending_human_review`, whether or not the automated flags detect issues. Numeric mismatches, empty answers, and truncation can be flagged, but **automatic heuristics are not proof of faithfulness**.

```bash
# Inspect the actual model outputs. This requires running the teacher first.
python3 - <<'PY'
import json
for line in open('output/teacher_outputs.jsonl', encoding='utf-8'):
    row = json.loads(line)
    print(row['id'], row['question'], '\n=>', row['teacher_answer'], '\nFLAGS:', row['review_flags'], '\n')
PY

# Once you have read, checked source factuality, and approved the chosen IDs:
printf 'FAQ001\nFAQ002\n' > output/approved_ids.txt
python3 src/export_reviewed.py --approved-ids output/approved_ids.txt
```

`output/approved_sft.jsonl` contains standard chat-style `messages`, e.g.:

```json
{"messages":[{"role":"user","content":"Example FAQ question"},{"role":"assistant","content":"An approved, real teacher-generated answer"}]}
```

That line above only documents the schema; **no actual teacher answers are included** until you run inference. Remove any entry failing human QA before approval. Split into training/validation/test sets in a future phase, to prevent leakage.

## 6 — Tests (run without downloading the model)

```bash
python3 -m unittest discover -s tests -v
```

The test calls are mocked: they do not run GPT-OSS-120B or incur EC2 inference costs.

## 7 — Upload *code* to GitHub

Run from the project folder; create an empty private GitHub repository named `gpt-oss-teacher-faq` first:

```bash
git init
git branch -M main
git add README.md .gitignore src scripts tests data/sample_faq.csv
git commit -m "Add Graviton GPT-OSS-120B teacher FAQ batch pipeline"
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/gpt-oss-teacher-faq.git
git push -u origin main
```

You must authenticate with GitHub (e.g. credential manager, token, or SSH). `.gitignore` excludes `models/`, `output/`, logs, `.env`, credentials. **Do not commit copyrighted/customer FAQ content without authorization**; replace demo data if you need a public repo.

## Caveats

- CPU-only Graviton is selected because that is the requirement, **not because it is necessarily the fastest or cheapest** method to generate 120B outputs.
- `gpt-oss-120b` uses the Harmony chat format; `llama.cpp` reads the embedded model template with `--jinja`.
- The checkpoint and `llama.cpp` APIs may change; pin exact model revision and llama.cpp commit for a reproducible benchmark / production run.
- Generated data is not proof of correctness; verify against trusted source before student training.
- Stop/terminate the EC2 instance when finished to control cost. Data on EBS may incur charges until volumes are deleted.

## References

- [AWS Graviton llama.cpp guidance](https://aws.github.io/graviton/machinelearning/llama.cpp.html)
- [llama.cpp gpt-oss guide](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [GGUF checkpoint repository](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF)
- [OpenAI gpt-oss documentation](https://github.com/openai/gpt-oss)
