#!/bin/bash
set -euo pipefail

WORKDIR="/home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw"
CE_DIR="$WORKDIR/compute-eval"
GENERATOR_PY="$WORKDIR/cutile-eval-kit/scripts/compute-eval-opencode-cutile-gpt52.py"
OPENCODE_BIN=/home/scratch.huizim_coreai/workdir/seedbot/workspace/opencode-cli/node_modules/opencode-linux-x64/bin/opencode
CE_PY="$CE_DIR/.venv/bin/python"
SKILL_MD="$WORKDIR/eval/cutile/cutile-minimal-skill.md"
TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
RUN_ROOT="$WORKDIR/runs/cutile-eval-control-$TIMESTAMP"

export OPENAI_API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"

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

# Fresh OpenCode state dirs to avoid stale session DB
export XDG_DATA_HOME="$RUN_ROOT/.xdg/data"
export XDG_CONFIG_HOME="$RUN_ROOT/.xdg/config"
export XDG_STATE_HOME="$RUN_ROOT/.xdg/state"
mkdir -p "$XDG_DATA_HOME" "$XDG_CONFIG_HOME" "$XDG_STATE_HOME"

mkdir -p "$RUN_ROOT"
echo "=== Control (baseline) ==="
echo "RUN_ROOT=$RUN_ROOT"
echo "SKILL_MD=$SKILL_MD"

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

echo "=== Control done ==="
