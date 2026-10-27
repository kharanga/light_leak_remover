#!/usr/bin/env bash
# Create the project virtualenv and install everything the notebook needs.
#
#   bash setup_env.sh                 # creates ./.env (or updates it if it already exists)
#   VENV=myenv bash setup_env.sh      # use another folder
#   PYTHON=python3.12 bash setup_env.sh
#
# Afterwards: source .env/bin/activate
set -euo pipefail

cd "$(dirname "$0")"
VENV="${VENV:-.env}"
PYTHON="${PYTHON:-python3}"
KERNEL_NAME="${KERNEL_NAME:-leakfix}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: $PYTHON not found. Install Python 3.9 or newer, or set PYTHON=/path/to/python" >&2
  exit 1
fi
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || {
  echo "error: $("$PYTHON" --version) is too old; Python 3.9 or newer is required" >&2
  exit 1
}

if [ ! -x "$VENV/bin/python" ]; then
  echo "creating virtualenv in $VENV"
  "$PYTHON" -m venv "$VENV"
else
  echo "updating existing virtualenv in $VENV"
fi

PIP="$VENV/bin/python -m pip"
$PIP install --upgrade pip

# core dependencies
$PIP install -r requirements.txt
echo "installing the leakfix package (editable)..."
$PIP install --upgrade setuptools wheel
$PIP install --no-build-isolation -e .

echo "installing notebook extras: ultralytics pulls in PyTorch (about 1 GB), this can take a few minutes..."

# notebooks: YOLO leak detector, Gemini, Jupyter kernel
$PIP install ultralytics "google-genai>=1.0" ipykernel

# register the venv as a Jupyter kernel so notebooks can select it by name
"$VENV/bin/python" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python (leakfix)"

"$VENV/bin/python" - <<'EOF'
import importlib.util
for name in ("numpy", "cv2", "scipy", "skimage", "matplotlib", "ultralytics", "google.genai", "leakfix"):
    found = importlib.util.find_spec(name) is not None
    print(f"  {name:12} {'ok' if found else 'MISSING'}")
EOF

echo
echo "done. Activate with:  source $VENV/bin/activate"
echo "In Jupyter / VS Code pick the kernel 'Python (leakfix)'."
echo "For the Gemini steps, set GEMINI_API_KEY in your shell (e.g. in ~/.zshrc)."
