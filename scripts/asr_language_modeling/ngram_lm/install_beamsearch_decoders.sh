#!/usr/bin/env bash
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

NEMO_PATH=/workspace/nemo
if [ "$#" -eq 1 ]; then
  NEMO_PATH=$1
fi
KENLM_MAX_ORDER=10

if [ -d "$NEMO_PATH" ]; then
  echo "The folder '$NEMO_PATH' exists."
else
  echo "Error: The folder '$NEMO_PATH' does not exist. Specify it as a first command line positional argument!"
  exit 1
fi

# ── System dependencies ──────────────────────────────────────────────────────
echo "==> Installing system dependencies..."
if [ "$(id -u)" -eq 0 ]; then
  apt-get update -qq
  apt-get upgrade -y -qq
  apt-get install -y -qq swig liblzma-dev libboost-all-dev cmake git wget build-essential
  rm -rf /var/lib/apt/lists/*
else
  sudo apt-get update -qq
  sudo apt-get upgrade -y -qq
  sudo apt-get install -y -qq swig liblzma-dev libboost-all-dev cmake git wget build-essential
  sudo rm -rf /var/lib/apt/lists/*
fi

export BOOST_ROOT=/usr

# ── Move into NeMo path ───────────────────────────────────────────────────────
cd "$NEMO_PATH"

# ── Clone & prepare OpenSeq2Seq decoders ─────────────────────────────────────
echo "==> Setting up OpenSeq2Seq decoders..."
if [ -d "decoders" ]; then
  echo "    'decoders' directory already exists, skipping clone."
else
  git clone https://github.com/NVIDIA/OpenSeq2Seq
  cd OpenSeq2Seq
  git checkout ctc-decoders
  cd ..
  mv OpenSeq2Seq/decoders "$NEMO_PATH/"
  rm -rf OpenSeq2Seq
fi

cd "$NEMO_PATH/decoders"

if [ -f "$NEMO_PATH/scripts/installers/setup_os2s_decoders.py" ]; then
  cp "$NEMO_PATH/scripts/installers/setup_os2s_decoders.py" ./setup.py
else
  echo "Warning: setup_os2s_decoders.py not found — skipping copy."
fi

# ── Build OpenFST ─────────────────────────────────────────────────────────────
echo "==> Building OpenFST..."
if [ ! -d "openfst-1.6.3" ]; then
  wget -q https://github.com/kkm000/openfst/archive/refs/tags/win/1.6.3.1.tar.gz -O openfst.tar.gz
  tar -xzf openfst.tar.gz
  mv openfst-win-1.6.3.1 openfst-1.6.3
  rm -f openfst.tar.gz
fi

if [ ! -f "openfst-1.6.3/src/lib/.libs/libfst.so" ]; then
  cd openfst-1.6.3
  ./configure --enable-static --enable-shared --enable-far --enable-ngram-fsts
  make -j"$(nproc)"
  cd ..
else
  echo "    OpenFST already built, skipping."
fi

# ── Clone & build KenLM ───────────────────────────────────────────────────────
echo "==> Setting up KenLM..."
if [ ! -d "kenlm" ]; then
  git clone https://github.com/kpu/kenlm kenlm
else
  echo "    kenlm directory already exists, skipping clone."
fi

mkdir -p kenlm/build
cd kenlm/build

if [ ! -f "Makefile" ] && [ ! -f "build.ninja" ]; then
  cmake .. \
    -DKENLM_MAX_ORDER=$KENLM_MAX_ORDER \
    -DCMAKE_BUILD_TYPE=Release
fi

make -j"$(nproc)"
cd ../..   # back to decoders/

export KENLM_ROOT="$NEMO_PATH/decoders/kenlm"
export KENLM_LIB="$NEMO_PATH/decoders/kenlm/build/bin"

echo "==> Installing KenLM Python bindings..."
cd "$NEMO_PATH/decoders/kenlm"
python setup.py install --max_order=$KENLM_MAX_ORDER
cd "$NEMO_PATH/decoders"

# ── Build ctc_decoders ────────────────────────────────────────────────────────
echo "==> Building ctc_decoders..."
if [ ! -f "setup.py" ]; then
  echo "Error: setup.py not found in $NEMO_PATH/decoders. Cannot build ctc_decoders."
  exit 1
fi

# Ensure KenLM headers are findable
export CPLUS_INCLUDE_PATH="$KENLM_ROOT:$KENLM_ROOT/lm:${CPLUS_INCLUDE_PATH:-}"

python setup.py build_ext --inplace

# ── Install Flashlight Text ───────────────────────────────────────────────────
echo "==> Installing flashlight-text..."
if [ ! -d "text" ]; then
  git clone https://github.com/flashlight/text
fi

cd text
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall
cd ..

echo ""
echo "All done! ctc_decoders and flashlight-text are installed."