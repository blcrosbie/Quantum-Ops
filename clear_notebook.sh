#!/usr/bin/env bash
set -euo pipefail

if command -v jupyter >/dev/null 2>&1; then
    JUPYTER_CMD=(jupyter nbconvert)
elif command -v python >/dev/null 2>&1 && python -c "import nbconvert" >/dev/null 2>&1; then
    JUPYTER_CMD=(python -m nbconvert)
elif command -v py >/dev/null 2>&1 && py -c "import nbconvert" >/dev/null 2>&1; then
    JUPYTER_CMD=(py -m nbconvert)
elif command -v python >/dev/null 2>&1 && python -c "import jupyter" >/dev/null 2>&1; then
    JUPYTER_CMD=(python -m jupyter nbconvert)
elif command -v py >/dev/null 2>&1 && py -c "import jupyter" >/dev/null 2>&1; then
    JUPYTER_CMD=(py -m jupyter nbconvert)
else
    echo "Error: nbconvert/jupyter not found. Install with: python -m pip install nbconvert" >&2
    exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root/use-cases"

find . -name "*.ipynb" -print0 | while read -d $'\0' file
do
    "${JUPYTER_CMD[@]}" --clear-output --inplace "$file"
done
