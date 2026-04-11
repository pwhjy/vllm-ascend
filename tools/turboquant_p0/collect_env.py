import argparse
import os
import platform
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import REPO_ROOT, detect_version, ensure_dir, maybe_run_command_to_file, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect TurboQuant P0 environment snapshot.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workspace-root", default=str(REPO_ROOT.parent))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    workspace_root = Path(args.workspace_root)
    vllm_root = workspace_root / "vllm"
    vllm_ascend_root = workspace_root / "vllm-ascend"

    python_exec = sys.executable
    maybe_run_command_to_file(
        [python_exec, "collect_env.py"],
        output_dir / "collect_env.txt",
        cwd=vllm_ascend_root,
    )
    maybe_run_command_to_file(
        [python_exec, "-m", "pip", "freeze"],
        output_dir / "pip_freeze.txt",
        cwd=vllm_ascend_root,
    )
    maybe_run_command_to_file(
        ["git", "-C", str(vllm_root), "rev-parse", "HEAD"],
        output_dir / "vllm_git_head.txt",
    )
    maybe_run_command_to_file(
        ["git", "-C", str(vllm_ascend_root), "rev-parse", "HEAD"],
        output_dir / "vllm_ascend_git_head.txt",
    )
    maybe_run_command_to_file(
        ["npu-smi", "info"],
        output_dir / "npu_smi.txt",
    )

    summary = {
        "python_executable": python_exec,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "cwd": os.getcwd(),
        "workspace_root": str(workspace_root),
        "versions": {
            "torch": detect_version("torch"),
            "torch-npu": detect_version("torch-npu"),
            "vllm": detect_version("vllm"),
            "vllm-ascend": detect_version("vllm-ascend"),
            "transformers": detect_version("transformers"),
            "datasets": detect_version("datasets"),
        },
        "env_vars": {
            "VLLM_USE_V2_MODEL_RUNNER": os.environ.get("VLLM_USE_V2_MODEL_RUNNER"),
            "VLLM_USE_MODELSCOPE": os.environ.get("VLLM_USE_MODELSCOPE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "HCCL_CONNECT_TIMEOUT": os.environ.get("HCCL_CONNECT_TIMEOUT"),
        },
    }
    write_json(output_dir / "env_summary.json", summary)


if __name__ == "__main__":
    main()
