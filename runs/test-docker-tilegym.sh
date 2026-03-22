#!/bin/bash
set -euo pipefail

WORKDIR="/home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw"
IMAGE_NAME="local/compute-eval-tilegym:13.1.0"
BASE_IMAGE="local/compute-eval-python:13.1.0"
IMAGE_TAR="$WORKDIR/runs/compute-eval-python-13.1.0.tar"

# Ensure base image exists
if ! docker image inspect "$BASE_IMAGE" &>/dev/null; then
  if [[ -f "$IMAGE_TAR" ]]; then
    echo "Loading base Docker image..."
    docker load -i "$IMAGE_TAR"
  else
    echo "Building base Docker image..."
    cd "$WORKDIR/compute-eval"
    docker build -f docker/Dockerfile.python-cuda13 -t "$BASE_IMAGE" .
    cd "$WORKDIR"
  fi
fi

# Build TileGym image
echo "=== Building TileGym Docker image ==="
cd "$WORKDIR"
docker build -f eval/cutile/Dockerfile.tilegym -t "$IMAGE_NAME" .
echo "Image built: $IMAGE_NAME"

echo ""
echo "=== Test 1: Check TileGym is installed ==="
docker run --rm "$IMAGE_NAME" python -c "import tilegym; print(f'TileGym OK: {tilegym.__version__}')"

echo ""
echo "=== Test 2: GPU + cuTile ==="
docker run --rm --gpus all "$IMAGE_NAME" python -c "
import torch, cuda.tile as ct
print(f'GPU: {torch.cuda.get_device_name(0)}, CC: {torch.cuda.get_device_capability(0)}')
print(f'cuTile: OK')
"

echo ""
echo "=== Test 3: Run softmax test ==="
docker run --rm --gpus all \
  -e CUDA_TILE_CACHE_DIR=/tmp/cutile-cache \
  "$IMAGE_NAME" \
  python -m pytest /opt/tilegym/tests/ops/test_softmax.py \
    -x -v -p no:cacheprovider --quick-run
SOFTMAX_EXIT=$?
echo "Softmax exit code: $SOFTMAX_EXIT"

echo ""
echo "=== Test 4: Run matmul test ==="
docker run --rm --gpus all \
  -e CUDA_TILE_CACHE_DIR=/tmp/cutile-cache \
  "$IMAGE_NAME" \
  python -m pytest /opt/tilegym/tests/ops/test_matmul.py \
    -x -v -p no:cacheprovider --quick-run
MATMUL_EXIT=$?
echo "Matmul exit code: $MATMUL_EXIT"

echo ""
echo "=== All tests done ==="
