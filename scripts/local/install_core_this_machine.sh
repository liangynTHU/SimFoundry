#!/usr/bin/env bash
# Core SimFoundry / OmniGibson setup for this machine.
#
# This intentionally installs only the `simfoundry` environment needed for
# Pipeline C and the OXE replay tools.  Hunyuan / Any6D / DA3 and model
# checkpoints are separate because they are large and require external model
# credentials for a full A-pipeline reconstruction.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${CONDA_ROOT:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_ROOT="$(conda info --base)"
    elif [[ -x /jizhicfs/peterrao/miniconda3/bin/conda ]]; then
        CONDA_ROOT=/jizhicfs/peterrao/miniconda3
    else
        echo "Set CONDA_ROOT to a Conda installation." >&2
        exit 2
    fi
fi
CONDA_EXE="${CONDA_ROOT}/bin/conda"
PROXY="${SIMFOUNDRY_HTTP_PROXY:-}"
ENV_NAME="${SIMFOUNDRY_ENV_NAME:-simfoundry}"

source "${CONDA_ROOT}/bin/activate"

# This Miniconda has Conda's libmamba solver but no standalone `mamba` binary.
export PATH="${REPO_DIR}/scripts/local:${CONDA_ROOT}/bin:${PATH}"
export SIMFOUNDRY_CONDA_EXE="${CONDA_EXE}"

if [[ -n "${PROXY}" ]]; then
    export http_proxy="${http_proxy:-${PROXY}}"
    export https_proxy="${https_proxy:-${PROXY}}"
    export HTTP_PROXY="${HTTP_PROXY:-${PROXY}}"
    export HTTPS_PROXY="${HTTPS_PROXY:-${PROXY}}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE:-1}"
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_DATA_PATH="${REPO_DIR}/deps/BEHAVIOR-1K/datasets"

cd "${REPO_DIR}"

if ! "${CONDA_EXE}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    # The upstream installer can stop at its final asset-download step if pip
    # leaves two charset-normalizer versions behind.  Preserve its log and
    # continue to the deterministic repair/finalization block below.
    set +e
    bash scripts/installation/install_simfoundry.sh \
        --project-root "${REPO_DIR}" \
        --env-name "${ENV_NAME}" \
        --cuda-version 12.8 \
        --cuda-arch-list "${TORCH_CUDA_ARCH_LIST}" \
        --default \
        2>&1 | tee logs/install_simfoundry_core.log
    installer_rc=${PIPESTATUS[0]}
    set -e
    echo "Upstream installer exit code: ${installer_rc}"
fi

PY="${CONDA_ROOT}/envs/${ENV_NAME}/bin/python"
if [[ ! -x "${PY}" ]]; then
    echo "ERROR: environment ${ENV_NAME} was not created successfully." >&2
    exit 1
fi

# Repair the duplicate compiled/pure-Python charset-normalizer mix observed on
# this host after the BEHAVIOR / Isaac dependency transaction.
if ! "${PY}" -c "import charset_normalizer, requests" >/dev/null 2>&1; then
    "${PY}" -m pip uninstall -y charset-normalizer || true
    "${PY}" -m pip uninstall -y charset-normalizer || true
    "${PY}" -m pip install --no-cache-dir --force-reinstall "charset-normalizer==3.3.2"
fi

# Runtime libraries used by Isaac Sim's MaterialX stack and Vulkan diagnostics.
"${CONDA_EXE}" install -n "${ENV_NAME}" -c conda-forge \
    xorg-libxt libglu vulkan-tools -y

# Required by SimFoundry's pose / retrieval utilities.
if ! "${PY}" -c "import faiss; assert hasattr(faiss, 'StandardGpuResources')" >/dev/null 2>&1; then
    "${CONDA_EXE}" install -n "${ENV_NAME}" -c pytorch "faiss-gpu=1.12" -y
fi

# Packages referenced by the C-pipeline policy / LeRobot paths.
"${PY}" -m pip install \
    "openpi-client==0.1.2" \
    "torchcodec==0.5.0" \
    "gymnasium==0.29.1"

# Public OmniGibson robot assets.
"${PY}" -c \
    "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()"

ROBOT_ASSET_ROOT="${OMNIGIBSON_DATA_PATH}/omnigibson-robot-assets"
ROBOTIQ_USD="${ROBOT_ASSET_ROOT}/models/franka/franka_robotiq/usd/franka_robotiq.usda"
if [[ ! -f "${ROBOTIQ_USD}" ]]; then
    "${PY}" - "${ROBOT_ASSET_ROOT}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="behavior-1k/omnigibson-robot-assets",
    repo_type="dataset",
    allow_patterns=["models/franka/franka_robotiq/**"],
    local_dir=sys.argv[1],
)
PY
fi

"${PY}" - <<'PY'
import faiss
import lerobot
import numpy
import omnigibson
import simfoundry
import torch

print("simfoundry", simfoundry.__file__)
print("omnigibson", omnigibson.__version__)
print("lerobot", lerobot.__version__)
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("faiss_gpus", faiss.get_num_gpus())
PY
