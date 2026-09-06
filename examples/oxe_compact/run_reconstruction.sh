#!/usr/bin/env bash
# Run SimFoundry Pipeline A on one bundled compact OXE/DROID camera stream.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="${REPO_DIR}/scripts/local:${PATH}"
EPISODE_INDEX="${1:-3}"
if [[ $# -gt 0 ]]; then
    shift
fi
printf -v EPISODE_NAME 'droid_episode_%06d' "${EPISODE_INDEX}"
EPISODE_DIR="${REPO_DIR}/examples/oxe_compact/${EPISODE_NAME}"
VIDEO_PATH="${EPISODE_DIR}/reconstruction_input.mp4"

if [[ ! -f "${EPISODE_DIR}/trajectory.npz" || ! -e "${VIDEO_PATH}" ]]; then
    echo "Bundled episode not found: ${EPISODE_DIR}" >&2
    echo "Available indices: 0, 3, 4, 5, 8" >&2
    exit 2
fi

exec bash "${REPO_DIR}/scripts/pipeline/A_reconstruction/run.sh" \
    --scene-name "oxe_${EPISODE_NAME}" \
    --root-dir "${REPO_DIR}/Data" \
    --video-fpath "${VIDEO_PATH}" \
    "$@"
