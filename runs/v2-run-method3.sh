#!/bin/bash
# V2 Evaluation: Method 3 — Method 2 skills + targeted validation/precision skills
# Based on failure analysis of 13 test-gap tasks
set -euo pipefail

WORKDIR="/home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw"
SCRIPT="$WORKDIR/cutile-synth-eval-kit/scripts/compute-eval-run-cutile-opencode-opus46-synthv2.sh"

export CE_DIR="/home/scratch.huizim_coreai/workdir/seedbot/compute-eval-api"
export SYNTH_ROOT="/home/scratch.huizim_coreai/workdir/compute-eval-synth/compute_eval_synth_v2"
export RUN_ROOT="$WORKDIR/runs/v2-method3-$(date -u +%Y%m%d-%H%M%S)"

# Single merged skill dir: base + Method 2 evolved + targeted validation/precision rules
export CUTILE_SKILL_DIRS="$WORKDIR/skills/cutile-python-method3"

export EVAL_IMAGE_TAR="$WORKDIR/runs/compute-eval-python-13.1.0.tar"

# Load OpenCode agent image from tar
OPENCODE_TAR="$WORKDIR/runs/opencode-agent-cutile-synthv2.tar"
if ! docker image inspect local/opencode-agent:cutile-synthv2 &>/dev/null; then
  echo "Loading OpenCode agent image..."
  docker load -i "$OPENCODE_TAR"
fi

export OPENAI_API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
export MAX_WORKERS=1

echo "=== V2 Method 3 (targeted fixes) ==="
echo "RUN_ROOT=$RUN_ROOT"
echo "SKILL_DIRS=$CUTILE_SKILL_DIRS"

exec bash "$SCRIPT"
