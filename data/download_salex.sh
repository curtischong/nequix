#!/bin/bash

# sAlex only (the MP-compatible subsampled Alexandria split shipped with OMat24);
# use download_omat.sh to also fetch the full OMat24 train/val tarballs.

# configure data root as first arg or DATA_ROOT env var (defaults to ./data)
DATA_ROOT="${1:-${DATA_ROOT:-data}}"

# sAlex train files
mkdir -p "$DATA_ROOT/salex/train/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex/train.tar.gz -P "$DATA_ROOT/salex/train/"
tar -xf "$DATA_ROOT/salex/train/train.tar.gz" -C "$DATA_ROOT/salex/train/"

# sAlex val files
mkdir -p "$DATA_ROOT/salex/val/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex/val.tar.gz -P "$DATA_ROOT/salex/val/"
tar -xf "$DATA_ROOT/salex/val/val.tar.gz" -C "$DATA_ROOT/salex/val/"
