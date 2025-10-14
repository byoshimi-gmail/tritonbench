import os
import subprocess
import sys

from pathlib import Path

REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
CUTLASS_PATH = REPO_PATH.joinpath("submodules", "cutlass")


def test_cutlass():
    cmd = [
        sys.executable,
        "-c",
        "import cutlass_cppgen",
    ]
    subprocess.check_call(cmd)

def install_cutlass():
    command = ["pip", "install", "-e", "."]
    subprocess.check_call(command, cwd=CUTLASS_PATH)
