#!/bin/bash
set -euo pipefail

WORKDIR="/home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw"
SCRIPT="$WORKDIR/cutile-synth-eval-kit/scripts/compute-eval-run-cutile-opencode-opus46-synthv2.sh"

export CE_DIR="/home/scratch.huizim_coreai/workdir/seedbot/compute-eval-api"
export SYNTH_ROOT="/home/scratch.huizim_coreai/workdir/compute-eval-synth/compute_eval_synth_v2"
export RUN_ROOT="$WORKDIR/runs/v2-synthesized-$(date -u +%Y%m%d-%H%M%S)"

# Synthesized skill dir (from run-synthesis-b200.sh)
export CUTILE_SKILL_DIRS="$WORKDIR/eval/cutile/synthesis-results/cutile-python-synthesized"

export EVAL_IMAGE_TAR="$WORKDIR/runs/compute-eval-python-13.1.0.tar"

# Load OpenCode agent image
OPENCODE_TAR="$WORKDIR/runs/opencode-agent-cutile-synthv2.tar"
if ! docker image inspect local/opencode-agent:cutile-synthv2 &>/dev/null; then
  echo "Loading OpenCode agent image..."
  docker load -i "$OPENCODE_TAR"
fi

export OPENAI_API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
export MAX_WORKERS=1

echo "=== V2 Synthesized (debug journey skills) ==="
echo "RUN_ROOT=$RUN_ROOT"
echo "SKILL_DIRS=$CUTILE_SKILL_DIRS"

exec bash "$SCRIPT"
