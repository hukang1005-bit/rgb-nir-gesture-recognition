#!/usr/bin/env python3
"""Create the local environment once, then run the standalone experiment."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def call(args):
    print(" ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=str(ROOT), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-only", action="store_true")
    args, remaining = parser.parse_known_args()
    if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
        raise SystemExit("请安装64位Python 3.10、3.11或3.12，再运行本入口。")
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if not args.setup_only:
        p = Path(config.get("dataset_root", "")).expanduser()
        p = p if p.is_absolute() else ROOT / p
        if not p.is_dir() or not (p / "rgb").is_dir() or not (p / "tof").is_dir():
            raise SystemExit(
                "请先修改config.json中的dataset_root，指向包含rgb/和tof/的Briareo目录。"
            )
    env = ROOT / ".venv"
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    req = ROOT / "requirements.txt"
    stamp = env / "gray_environment_ready.json"
    digest = hashlib.sha256(req.read_bytes()).hexdigest()
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(str(env))
    ready = json.loads(stamp.read_text()) if stamp.exists() else {}
    if ready.get("requirements_sha256") != digest:
        call([python, "-m", "pip", "install", "--upgrade", "pip"])
        torch_args = [python, "-m", "pip", "install", "torch==2.8.0"]
        if sys.platform in ["win32", "linux"]:
            torch_args += ["--index-url", "https://download.pytorch.org/whl/cpu"]
        call(torch_args)
        call([python, "-m", "pip", "install", "-r", req])
        stamp.write_text(
            json.dumps({"requirements_sha256": digest, "platform": platform.platform()}, indent=2)
        )
    call([python, "run.py", "self-test"])
    if not args.setup_only:
        call([python, "run.py"] + (remaining or ["run"]))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit("启动失败：" + str(e))
