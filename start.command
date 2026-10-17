#!/bin/bash
set -e
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$task_dir"
if command -v python3.11 >/dev/null 2>&1; then
  python3.11 bootstrap.py "$@"
elif command -v python3.10 >/dev/null 2>&1; then
  python3.10 bootstrap.py "$@"
else
  python3 bootstrap.py "$@"
fi
