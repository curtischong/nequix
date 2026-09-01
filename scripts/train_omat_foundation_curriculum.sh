#!/usr/bin/env bash
set -euo pipefail

# Two-stage OMat24 foundation-model curriculum:
#   1. two epochs with an auxiliary direct-force head
#   2. two epochs with conservative (energy-gradient) forces
#
# Each stage resumes from its own full training-state checkpoint when present.
# The handoff between stages is weights-only, so stage two intentionally starts
# with a fresh optimizer and learning-rate schedule.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

for data_path in data/omat/train.atp data/omat/val.atp; do
    if [[ ! -f "$data_path" ]]; then
        echo "Missing required OMat24 data: $data_path" >&2
        exit 1
    fi
done

mkdir -p checkpoints

UV_BIN="${UV_BIN:-uv}"
DIRECT_CHECKPOINT="checkpoints/nequix-omat-foundation-direct.nqx"
CONSERVATIVE_STATE="checkpoints/nequix-omat-foundation-conservative.pkl"

echo "Stage 1/2: direct-force pre-training (2 epochs)"
"$UV_BIN" run --no-sync train nequix-omat-foundation-direct

if [[ ! -f "$DIRECT_CHECKPOINT" && ! -f "$CONSERVATIVE_STATE" ]]; then
    echo "Stage one did not produce the weights checkpoint: $DIRECT_CHECKPOINT" >&2
    echo "If its state says training is complete, remove that stage-one state and rerun." >&2
    exit 1
fi

echo "Stage 2/2: conservative fine-tuning (2 epochs, fresh optimizer)"
"$UV_BIN" run --no-sync train nequix-omat-foundation-conservative

echo "Curriculum complete. Best conservative model:"
echo "  checkpoints/nequix-omat-foundation-conservative.nqx"
