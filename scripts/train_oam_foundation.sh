#!/usr/bin/env bash
set -euo pipefail

# Stage three of the foundation-model curriculum (the eSEN OAM recipe): one
# conservative epoch on sAlex + 8x MPtrj, fine-tuned from the stage-two OMat24
# conservative checkpoint. This stage moves the model onto the MP-compatible
# energy reference required by Matbench Discovery; OMat24 itself never appears
# in this mix because its DFT settings are incompatible with MPtrj/sAlex/WBM.
#
# Kept separate from train_omat_foundation_curriculum.sh so it can be prepared
# and launched while the OMat stages are still running.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

for data_path in data/mptrj.atp data/salex/train.atp data/salex/val.atp; do
    if [[ ! -f "$data_path" ]]; then
        echo "Missing required OAM data: $data_path" >&2
        echo "Build sAlex with: bash data/download_salex.sh, then" >&2
        echo "  uv run python scripts/preprocess_ase_db.py data/salex/train data/salex/train.atp --n_workers 32" >&2
        echo "  uv run python scripts/preprocess_ase_db.py data/salex/val data/salex/val.atp --n_workers 32" >&2
        exit 1
    fi
done

CONSERVATIVE_CHECKPOINT="checkpoints/nequix-omat-foundation-conservative.nqx"
OAM_STATE="checkpoints/nequix-oam-foundation.pkl"

if [[ ! -f "$CONSERVATIVE_CHECKPOINT" && ! -f "$OAM_STATE" ]]; then
    echo "Missing the stage-two checkpoint: $CONSERVATIVE_CHECKPOINT" >&2
    echo "Run scripts/train_omat_foundation_curriculum.sh first." >&2
    exit 1
fi

mkdir -p checkpoints

UV_BIN="${UV_BIN:-uv}"

echo "Stage 3/3: OAM fine-tuning (1 epoch on sAlex + 8x MPtrj, fresh optimizer)"
"$UV_BIN" run --no-sync train nequix-oam-foundation

echo "Curriculum complete. Best OAM model:"
echo "  checkpoints/nequix-oam-foundation.nqx"
