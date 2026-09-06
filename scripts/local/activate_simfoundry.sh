#!/usr/bin/env bash
# Source this file from an interactive shell:
#   source scripts/local/activate_simfoundry.sh

_SIMFOUNDRY_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -n "${CONDA_ROOT:-}" ]]; then
    _SIMFOUNDRY_CONDA_ROOT="${CONDA_ROOT}"
elif command -v conda >/dev/null 2>&1; then
    _SIMFOUNDRY_CONDA_ROOT="$(conda info --base)"
elif [[ -x /jizhicfs/peterrao/miniconda3/bin/conda ]]; then
    # Convenience fallback for the original H20 host.
    _SIMFOUNDRY_CONDA_ROOT=/jizhicfs/peterrao/miniconda3
else
    echo "Cannot find Conda. Set CONDA_ROOT before sourcing this file." >&2
    return 1 2>/dev/null || exit 1
fi

if [[ -f "${_SIMFOUNDRY_CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${_SIMFOUNDRY_CONDA_ROOT}/etc/profile.d/conda.sh"
else
    # shellcheck disable=SC1091
    source "${_SIMFOUNDRY_CONDA_ROOT}/bin/activate"
fi
conda activate "${SIMFOUNDRY_ENV_NAME:-simfoundry}"

export PATH="${_SIMFOUNDRY_REPO}/scripts/local:${PATH}"
export PYTHONPATH="${_SIMFOUNDRY_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export SIMFOUNDRY_CONDA_EXE="${_SIMFOUNDRY_CONDA_ROOT}/bin/conda"
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_DATA_PATH="${_SIMFOUNDRY_REPO}/deps/BEHAVIOR-1K/datasets"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "Activated ${CONDA_DEFAULT_ENV}"
echo "SimFoundry repo: ${_SIMFOUNDRY_REPO}"
echo "OmniGibson data: ${OMNIGIBSON_DATA_PATH}"
