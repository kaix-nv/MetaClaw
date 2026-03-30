#!/bin/bash
set -euo pipefail

WORKDIR="/home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw"
TILEGYM="$WORKDIR/TileGym"
BASE_SKILL="$WORKDIR/skills/cutile-python"
RESULTS="$WORKDIR/eval/cutile/synthesis-results"

export API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
export OPENAI_API_KEY="$API_KEY"

# Load Docker images
for img_name in "local/opencode-agent:cutile-synthv2" "local/compute-eval-tilegym:13.1.0"; do
  if ! docker image inspect "$img_name" &>/dev/null; then
    tar_name=$(echo "$img_name" | sed 's|local/||;s|:|-|').tar
    tar_path="$WORKDIR/runs/$tar_name"
    if [[ -f "$tar_path" ]]; then
      echo "Loading $img_name from tar..."
      docker load -i "$tar_path"
    else
      echo "WARNING: $img_name not found and no tar at $tar_path"
    fi
  fi
done

echo "=== Iterative Solver + Skill Synthesis ==="
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
  --max-steps 48

echo "=== Synthesis done ==="
echo "Merged skill: $RESULTS/cutile-python-synthesized/"
