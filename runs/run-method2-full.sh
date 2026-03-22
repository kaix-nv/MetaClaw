#!/bin/bash
set -euo pipefail

WORKDIR="/home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw"
TILEGYM="$WORKDIR/TileGym"
CE_DIR="$WORKDIR/compute-eval"
BASE_SKILL="$WORKDIR/eval/cutile/cutile-minimal-skill.md"
RESULTS="$WORKDIR/eval/cutile/results"
OPENCODE_BIN=/home/scratch.huizim_coreai/workdir/seedbot/workspace/opencode-cli/node_modules/opencode-linux-x64/bin/opencode
CE_PY="$CE_DIR/.venv/bin/python"
GENERATOR_PY="$WORKDIR/cutile-eval-kit/scripts/compute-eval-opencode-cutile-gpt52.py"
TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
RUN_ROOT="$WORKDIR/runs/cutile-eval-method2-$TIMESTAMP"

export OPENAI_API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"

echo "=== Method 2 Full Pipeline on B200: $(hostname) ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# --- Step 1: Setup environment ---
echo "=== Step 1: Setup ==="
# Use compute-eval venv which has proper Python environment
CE_VENV="$CE_DIR/.venv/bin"
if [[ -f "$CE_VENV/pip" ]]; then
  "$CE_VENV/pip" install openai 2>&1 | tail -3
  export PATH="$CE_VENV:$PATH"
else
  pip install openai 2>&1 | tail -3
fi

# Load TileGym Docker image for KernelTester (Phase B)
TILEGYM_IMAGE="local/compute-eval-tilegym:13.1.0"
TILEGYM_TAR="$WORKDIR/runs/compute-eval-tilegym-13.1.0.tar"
if ! docker image inspect "$TILEGYM_IMAGE" &>/dev/null; then
  if [[ -f "$TILEGYM_TAR" ]]; then
    echo "Loading TileGym Docker image..."
    docker load -i "$TILEGYM_TAR"
  else
    echo "ERROR: TileGym Docker image not found. Build it first:"
    echo "  docker build -f eval/cutile/Dockerfile.tilegym -t $TILEGYM_IMAGE ."
    echo "  docker save $TILEGYM_IMAGE -o $TILEGYM_TAR"
    exit 1
  fi
fi
echo "TileGym Docker image ready: $TILEGYM_IMAGE"

# Load compute-eval Docker image (Phase C evaluation)
CE_IMAGE="local/compute-eval-python:13.1.0"
CE_TAR="$WORKDIR/runs/compute-eval-python-13.1.0.tar"
if ! docker image inspect "$CE_IMAGE" &>/dev/null; then
  if [[ -f "$CE_TAR" ]]; then
    echo "Loading compute-eval Docker image..."
    docker load -i "$CE_TAR"
  fi
fi

cd "$WORKDIR"

# --- Step 2: Skill Evolution (Phase A + Phase B with GPU tests) ---
echo "=== Step 2: Skill Evolution ==="
rm -rf "$RESULTS/method-2-multiturn"

python eval/cutile/run_eval.py \
  --method multiturn \
  --tilegym-dir "$TILEGYM" \
  --base-skill "$BASE_SKILL" \
  --results-dir "$RESULTS" \
  --model aws/anthropic/bedrock-claude-opus-4-6 \
  --rounds 3

echo "Skills generated:"
ls "$RESULTS/method-2-multiturn/evolved-skills/"*/SKILL.md 2>/dev/null | grep -v base-skill || echo "  (none from Phase B)"

# --- Step 3: Generate Solutions with OpenCode ---
echo "=== Step 3: Generate Solutions ==="
SKILL_MD="$RESULTS/method-2-multiturn/cutile-skill-merged.md"

if [[ ! -f "$SKILL_MD" ]]; then
  echo "ERROR: Merged skill file not found at $SKILL_MD"
  exit 1
fi

# OpenCode config
export OPENCODE_CONFIG_CONTENT=$(python3 -c "
import json, os
config = {
    'model': 'nvidia/aws/anthropic/bedrock-claude-opus-4-6',
    'small_model': 'nvidia/aws/anthropic/bedrock-claude-opus-4-6',
    'provider': {
        'nvidia': {
            'name': 'nvidia',
            'npm': '@ai-sdk/openai-compatible',
            'models': {
                'aws/anthropic/bedrock-claude-opus-4-6': {
                    'name': 'nvidia/aws/anthropic/bedrock-claude-opus-4-6',
                    'id': 'aws/anthropic/bedrock-claude-opus-4-6'
                }
            },
            'options': {
                'apiKey': os.environ.get('OPENAI_API_KEY', ''),
                'baseURL': 'https://inference-api.nvidia.com/v1'
            }
        }
    }
}
print(json.dumps(config, separators=(',', ':')))
")

# Fresh OpenCode state
export XDG_DATA_HOME="$RUN_ROOT/.xdg/data"
export XDG_CONFIG_HOME="$RUN_ROOT/.xdg/config"
export XDG_STATE_HOME="$RUN_ROOT/.xdg/state"
mkdir -p "$XDG_DATA_HOME" "$XDG_CONFIG_HOME" "$XDG_STATE_HOME"
mkdir -p "$RUN_ROOT"

"$CE_PY" "$GENERATOR_PY" \
  --repo "$CE_DIR" \
  --skill-md "$SKILL_MD" \
  --opencode-bin "$OPENCODE_BIN" \
  --output-dir "$RUN_ROOT/output" \
  --run-root "$RUN_ROOT" \
  --release 2026-1 \
  --group cutile \
  --model "nvidia/aws/anthropic/bedrock-claude-opus-4-6" \
  --max-attempts 3

# --- Step 4: Evaluate ---
echo "=== Step 4: Evaluate ==="
IMAGE_NAME="local/compute-eval-python:13.1.0"
IMAGE_TAR="$WORKDIR/runs/compute-eval-python-13.1.0.tar"

if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
  if [[ -f "$IMAGE_TAR" ]]; then
    echo "Loading Docker image from tar..."
    docker load -i "$IMAGE_TAR"
  else
    echo "Building Docker image..."
    cd "$CE_DIR"
    docker build -f docker/Dockerfile.python-cuda13 -t "$IMAGE_NAME" .
    cd "$WORKDIR"
  fi
fi

export IMAGE_REGISTRY=local
SOL=$(ls "$RUN_ROOT/output/"*-solutions.tar.gz 2>/dev/null | head -1)

if [[ -z "$SOL" ]]; then
  echo "ERROR: No solutions datapack found"
  exit 1
fi

cd "$CE_DIR"
uv run python "$WORKDIR/runs/eval_patched.py" "$SOL"
mv 2026-1-*-graded-solutions.jsonl "$RUN_ROOT/" 2>/dev/null || true
cd "$WORKDIR"

# --- Step 5: Results ---
echo ""
echo "=== RESULTS ==="
graded=$(ls "$RUN_ROOT"/*-graded-solutions.jsonl 2>/dev/null | head -1)
if [[ -n "$graded" ]]; then
  python3 -c "
import json
with open('$graded') as f:
    entries = [json.loads(l) for l in f if l.strip()]
p = sum(1 for e in entries if e.get('passed'))
s = sum(1 for e in entries if e.get('skipped'))
f = len(entries) - p - s
print(f'  Method 2: passed={p} failed={f} skipped={s} total={len(entries)}')
"
else
  echo "  Method 2: NO RESULTS"
fi

echo ""
echo "=== Method 2 Full Pipeline Done ==="
echo "Run root: $RUN_ROOT"
echo "Skills: $RESULTS/method-2-multiturn/evolved-skills/"
echo "Graded: $graded"
