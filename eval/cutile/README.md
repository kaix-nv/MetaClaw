# cuTile Skill Evolution Evaluation

Compares one-shot skill extraction vs multi-turn conversational evolution
on cuTile kernel coding. Skills learned from TileGym, evaluated on compute-eval.

## Quick Start (local, no GPU)

```bash
# Method 1 only (direct extraction):
python eval/cutile/run_eval.py \
  --method direct \
  --tilegym-dir TileGym \
  --results-dir eval/cutile/results

# Method 2 (multi-turn, 2 rounds):
python eval/cutile/run_eval.py \
  --method multiturn \
  --tilegym-dir TileGym \
  --results-dir eval/cutile/results \
  --rounds 2

# Both methods + comparison:
python eval/cutile/run_eval.py \
  --method all \
  --tilegym-dir TileGym \
  --results-dir eval/cutile/results \
  --local-test
```

## Phase C (needs B200)

```bash
python eval/cutile/run_eval.py \
  --method evaluate \
  --eval-kit-dir cutile-eval-kit \
  --skills-from eval/cutile/results/method-2-multiturn/cutile-skill-merged.md
```

## Required Environment

- `OPENAI_API_KEY` or `AZURE_API_KEY`
- Python 3.10+ with MetaClaw dependencies
- B200 allocation for Phase C only
