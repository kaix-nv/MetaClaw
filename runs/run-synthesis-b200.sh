#!/bin/bash
# runs/run-synthesis-b200.sh
set -euo pipefail

WORKDIR="/home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw"
TILEGYM="$WORKDIR/TileGym"
BASE_SKILL="$WORKDIR/skills/cutile-python"
RESULTS="$WORKDIR/eval/cutile/synthesis-results"

export API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
export OPENAI_API_KEY="$API_KEY"

# Load Docker images
OPENCODE_IMAGE="local/opencode-agent:cutile-synthv2"
OPENCODE_TAR="$WORKDIR/runs/opencode-agent-cutile-synthv2.tar"
if ! docker image inspect "$OPENCODE_IMAGE" &>/dev/null; then
  if [[ -f "$OPENCODE_TAR" ]]; then
    echo "Loading OpenCode agent image..."
    docker load -i "$OPENCODE_TAR"
  else
    echo "ERROR: OpenCode image not found. Build it first."
    exit 1
  fi
fi

echo "=== Autoresearch-Style Skill Synthesis ==="
echo "TileGym: $TILEGYM"
echo "Base skill: $BASE_SKILL"
echo "Results: $RESULTS"

# Clean previous results
rm -rf "$RESULTS"

# Install openai if needed
pip install openai 2>/dev/null || true

cd "$WORKDIR"
python eval/cutile/run_skill_synthesis.py \
  --tilegym-dir "$TILEGYM" \
  --results-dir "$RESULTS" \
  --base-skill "$BASE_SKILL" \
  --model aws/anthropic/bedrock-claude-opus-4-6 \
  --timeout 600

echo "=== Done ==="
echo "Skills: $RESULTS/cutile-python-synthesized/"
ls "$RESULTS/cutile-python-synthesized/SKILL.md" 2>/dev/null && echo "Merged SKILL.md exists" || echo "WARNING: No merged skill"
