#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-lerobot}"
VENDOR_DIR="${PISTAR_RUNTIME_VENDOR_DIR:-/tmp/pistar_lerobot_vendor}"
LEROBOT_PACKAGE="${PISTAR_LEROBOT_PACKAGE:-${ROOT}/venv/lib/python3.11/site-packages/lerobot}"

if [[ ! -d "${LEROBOT_PACKAGE}" ]]; then
  echo "LeRobot package not found: ${LEROBOT_PACKAGE}" >&2
  echo "Set PISTAR_LEROBOT_PACKAGE=/path/to/lerobot if your venv path is different." >&2
  exit 2
fi

mkdir -p "${VENDOR_DIR}"
ln -sfn "${LEROBOT_PACKAGE}" "${VENDOR_DIR}/lerobot"

export PYTHONPATH="${VENDOR_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

exec conda run --no-capture-output -n "${CONDA_ENV}" \
  python "${ROOT}/control_your_robot/example/deploy/piper_pi05_rtc_rollout_collect.py" "$@"
