"""Result folders. Every experiment reads and writes below RESULTS_ROOT, in the same sub-folders as in our runs
(v2 ... v7). Default: the results/ folder of this repository. To use another folder, set the environment variable
VLA_GATE_RESULTS before starting Jupyter, e.g.  export VLA_GATE_RESULTS=~/smolvla_project/results"""
import os
_SETUP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_SETUP_DIR, "..", ".."))
RESULTS_ROOT = os.path.expanduser(os.environ.get("VLA_GATE_RESULTS", os.path.join(REPO_ROOT, "results")))
os.makedirs(RESULTS_ROOT, exist_ok=True)
print("results folder:", RESULTS_ROOT)
