#!/bin/bash
set -euo pipefail

WORKDIR="/home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw"
CE_DIR="$WORKDIR/compute-eval"
IMAGE_NAME="local/compute-eval-python:13.1.0"
IMAGE_TAR="$WORKDIR/runs/compute-eval-python-13.1.0.tar"

cd "$CE_DIR"

# Load Docker image from tar if not already present
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
  echo "Loading evaluator Docker image from tar..."
  docker load -i "$IMAGE_TAR"
fi
echo "Docker image ready: $IMAGE_NAME"

export IMAGE_REGISTRY=local

for condition in control method1 method2; do
  echo "=== Evaluating $condition ==="

  # Find latest run dir for this condition
  RUN_DIR=$(ls -td "$WORKDIR/runs/cutile-eval-${condition}-2026032"*/ 2>/dev/null | head -1)
  if [[ -z "$RUN_DIR" ]]; then
    echo "ERROR: No run dir found for $condition"
    continue
  fi

  SOL=$(ls "$RUN_DIR/output/"*-solutions.tar.gz 2>/dev/null | head -1)
  if [[ -z "$SOL" ]]; then
    echo "ERROR: No solutions found in $RUN_DIR/output/"
    continue
  fi

  echo "Run dir: $RUN_DIR"
  echo "Solutions: $SOL"

  # Use patched evaluator (fixes GPU detection on B200)
  uv run python "$WORKDIR/runs/eval_patched.py" "$SOL"

  # Move graded results to run dir
  mv 2026-1-*-graded-solutions.jsonl "$RUN_DIR/" 2>/dev/null || true
  echo "=== $condition done ==="
done

echo ""
echo "=== RESULTS ==="
for condition in control method1 method2; do
  RUN_DIR=$(ls -td "$WORKDIR/runs/cutile-eval-${condition}-2026032"*/ 2>/dev/null | head -1)
  graded=$(ls "$RUN_DIR"/*-graded-solutions.jsonl 2>/dev/null | head -1)
  if [[ -n "$graded" ]]; then
    python3 -c "
import json
with open('$graded') as f:
    entries = [json.loads(l) for l in f if l.strip()]
p = sum(1 for e in entries if e.get('passed'))
s = sum(1 for e in entries if e.get('skipped'))
f = len(entries) - p - s
print(f'  $condition: passed={p} failed={f} skipped={s} total={len(entries)}')
"
  else
    echo "  $condition: NO RESULTS"
  fi
done
