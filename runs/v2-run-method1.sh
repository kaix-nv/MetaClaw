#!/bin/bash
# V2 Evaluation: Method 1 — using cutile-synth-eval-kit (tests hidden, Docker isolated)
# Skill: default cutile-python + Method 1 evolved skills
set -euo pipefail

WORKDIR="/home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw"
SCRIPT="$WORKDIR/cutile-synth-eval-kit/scripts/compute-eval-run-cutile-opencode-opus46-synthv2.sh"

export CE_DIR="$WORKDIR/compute-eval"
export SYNTH_ROOT="$WORKDIR/compute-eval-synth/compute_eval_synth_v2"
export RUN_ROOT="$WORKDIR/runs/v2-method1-$(date -u +%Y%m%d-%H%M%S)"

# Default skill + Method 1 evolved skills (colon-separated)
export CUTILE_SKILL_DIRS="/home/scratch.huizim_coreai/workdir/cutile-skills/skills/cutile-python:$WORKDIR/eval/cutile/results/method-1-direct/evolved-skills/cutile-kernel-patterns:$WORKDIR/eval/cutile/results/method-1-direct/evolved-skills/cutile-attention-kernels"

export EVAL_IMAGE_TAR="$WORKDIR/runs/compute-eval-python-13.1.0.tar"
export OPENAI_API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
export MAX_WORKERS=1

echo "=== V2 Method 1 ==="
echo "RUN_ROOT=$RUN_ROOT"
echo "SKILL_DIRS=$CUTILE_SKILL_DIRS"

exec bash "$SCRIPT"
