"""Run evaluate_functional_correctness with CLI Docker backend for B200."""
import os
import sys

os.environ["COMPUTE_EVAL_DOCKER_BACKEND"] = "cli"
os.environ.setdefault("IMAGE_REGISTRY", "local")

from compute_eval.evaluation import evaluate_functional_correctness
from compute_eval.data.data_model import ReleaseVersion

solutions = sys.argv[1]
ce_dir = "/home/scratch.kaix_coreai/workspace/MemSkill/MetaClaw/compute-eval"

evaluate_functional_correctness(
    release=ReleaseVersion("2026-1"),
    solutions_datapack=solutions,
    problems_datapack_dir=os.path.join(ce_dir, "data", "releases"),
    mode="docker",
    k=1,
    n_workers=1,
    profile_mode=None,
)
