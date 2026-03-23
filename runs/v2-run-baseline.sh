#!/bin/bash
# V2 Evaluation: Baseline — using cutile-synth-eval-kit (tests hidden, Docker isolated)
# Skill: default cutile-python skill from huizim's workspace
set -euo pipefail

WORKDIR="/home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw"
SCRIPT="$WORKDIR/cutile-synth-eval-kit/scripts/compute-eval-run-cutile-opencode-opus46-synthv2.sh"

# Override paths to use our repos
export CE_DIR="$WORKDIR/compute-eval"
export SYNTH_ROOT="$WORKDIR/compute-eval-synth/compute_eval_synth_v2"
export RUN_ROOT="$WORKDIR/runs/v2-baseline-$(date -u +%Y%m%d-%H%M%S)"

# Use default skill (same as friend's baseline)
export CUTILE_SKILL_DIRS="/home/scratch.huizim_coreai/workdir/cutile-skills/skills/cutile-python"

# Eval image
export EVAL_IMAGE_TAR="$WORKDIR/runs/compute-eval-python-13.1.0.tar"

# API key
export OPENAI_API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"

# Single worker for deterministic ordering
export MAX_WORKERS=1

echo "=== V2 Baseline ==="
echo "RUN_ROOT=$RUN_ROOT"
echo "SKILL_DIRS=$CUTILE_SKILL_DIRS"

exec bash "$SCRIPT"
