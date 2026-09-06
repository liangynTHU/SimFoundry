#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIMFOUNDRY_CONDA_EXE="${SIMFOUNDRY_CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${SIMFOUNDRY_CONDA_EXE}" && -x /jizhicfs/peterrao/miniconda3/bin/conda ]]; then
    SIMFOUNDRY_CONDA_EXE=/jizhicfs/peterrao/miniconda3/bin/conda
fi
if [[ -z "${SIMFOUNDRY_CONDA_EXE}" || ! -x "${SIMFOUNDRY_CONDA_EXE}" ]]; then
    echo "Set SIMFOUNDRY_CONDA_EXE to a working conda executable." >&2
    exit 2
fi
RLDS_ENV="${RLDS_ENV:-rlds_env}"
SIM_ENV="${SIM_ENV:-simfoundry}"
LOCAL_DROID_ROOT=/apdcephfs_gy5/share_303588738/peterrao/data/vla-data/OXE/droid/1.4.0
OXE_DROID_ROOT="${OXE_DROID_ROOT:-}"
if [[ -z "${OXE_DROID_ROOT}" && -d "${LOCAL_DROID_ROOT}" ]]; then
    OXE_DROID_ROOT="${LOCAL_DROID_ROOT}"
fi
if [[ -z "${OXE_DROID_ROOT}" || ! -d "${OXE_DROID_ROOT}" ]]; then
    echo "Set OXE_DROID_ROOT to the local DROID TFDS builder directory." >&2
    echo "For bundled trajectories that need no TFDS data, see examples/oxe_compact/." >&2
    exit 2
fi
# Episode 28 has three consistent language annotations for a marker-to-container
# task, making it a clearer default demonstration than trajectories whose
# alternative language annotations disagree.
EPISODE_INDEX="${EPISODE_INDEX:-28}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/artifacts/oxe/droid_episode_$(printf '%06d' "${EPISODE_INDEX}")}"
BACKEND="${BACKEND:-pybullet}"

if [[ "${BACKEND}" != "pybullet" && "${BACKEND}" != "omnigibson" ]]; then
    echo "Unknown BACKEND=${BACKEND}; choose pybullet or omnigibson." >&2
    exit 2
fi

if [[ "${BACKEND}" == "omnigibson" ]]; then
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null || true)"
    if [[ "${SIMFOUNDRY_ALLOW_UNSUPPORTED_GPU:-0}" != "1" ]] \
        && [[ "${gpu_name}" =~ (A100|H100|H20) ]]; then
        cat >&2 <<EOF
Native OmniGibson was not started.

Detected GPU: ${gpu_name}
The current Isaac Sim renderer requires an RTX-capable GPU. NVIDIA's
requirements explicitly exclude non-RT-core datacenter GPUs such as A100/H100,
and this H20 host crashed during the local renderer smoke test.

Use the default BACKEND=pybullet on this node, or run BACKEND=omnigibson on an
RTX-capable node with NVIDIA graphics/Vulkan driver capabilities exposed.
Set SIMFOUNDRY_ALLOW_UNSUPPORTED_GPU=1 only to bypass this guard for debugging.
EOF
        exit 3
    fi
fi

CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 \
    "${SIMFOUNDRY_CONDA_EXE}" run -n "${RLDS_ENV}" python \
    "${REPO_DIR}/tools/oxe/extract_droid_trajectory.py" \
    --dataset-root "${OXE_DROID_ROOT}" \
    --episode-index "${EPISODE_INDEX}" \
    --output-dir "${OUTPUT_DIR}" \
    --overwrite

case "${BACKEND}" in
    pybullet)
        CUDA_VISIBLE_DEVICES="" "${SIMFOUNDRY_CONDA_EXE}" run -n "${SIM_ENV}" python \
            "${REPO_DIR}/tools/oxe/replay_droid_trajectory_in_pybullet.py" \
            "${OUTPUT_DIR}/trajectory.npz" \
            --output-dir "${OUTPUT_DIR}/pybullet_replay" \
            "$@"
        ;;
    omnigibson)
        export OMNI_KIT_ACCEPT_EULA=YES
        export OMNIGIBSON_DATA_PATH="${REPO_DIR}/deps/BEHAVIOR-1K/datasets"
        "${SIMFOUNDRY_CONDA_EXE}" run -n "${SIM_ENV}" python \
            "${REPO_DIR}/tools/oxe/replay_droid_trajectory_in_omnigibson.py" \
            "${OUTPUT_DIR}/trajectory.npz" \
            --output-dir "${OUTPUT_DIR}/omnigibson_replay" \
            "$@"
        ;;
esac
