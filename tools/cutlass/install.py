REPO_PATH = Path(os.path.abspath(__file__)).parent.parent.parent
CUTLASS_PATH = REPO_PATH.joinpath("submodules", "cutlass")


def test_cutlass():
    cmd = [
        sys.executable,
        "-c",
        "import cutlass_cppgen",
    ]
    subprocess.check_call(cmd, env=environ)

def install_cutlass():
    command = ["pip", "install", "-e", "."]
    subprocess.check_call(cmd, cwd=CUTLASS_PATH)
