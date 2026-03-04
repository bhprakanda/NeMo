#!/usr/bin/env bash
set -euo pipefail

NEMO_PATH="${1:-/kaggle/working/NeMo}"
DECODERS_DIR="$NEMO_PATH/decoders"
KENLM_MAX_ORDER=10

echo "==> NEMO_PATH=$NEMO_PATH"

if [ ! -d "$NEMO_PATH" ]; then
  echo "Error: '$NEMO_PATH' does not exist."
  exit 1
fi

# ── System dependencies ───────────────────────────────────────────────────────
echo "==> Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
  swig liblzma-dev libboost-all-dev cmake git wget build-essential \
  2>&1 | tail -5
export BOOST_ROOT=/usr

# ── Full clean of build artifacts ────────────────────────────────────────────
echo "==> Cleaning previous build artifacts..."
cd "$NEMO_PATH"
rm -rf OpenSeq2Seq
rm -rf "$DECODERS_DIR/openfst-win-1.6.3.1"
rm -rf "$DECODERS_DIR/openfst.tar.gz"
rm -rf "$DECODERS_DIR/kenlm"
rm -rf "$DECODERS_DIR/text"
rm -rf "$DECODERS_DIR/build"
rm -f  "$DECODERS_DIR/_swig_decoders"*.so 2>/dev/null || true

# ── OpenSeq2Seq decoders ──────────────────────────────────────────────────────
echo "==> Setting up OpenSeq2Seq decoders..."
cd "$NEMO_PATH"

# Always re-clone to ensure ThreadPool submodule is present
rm -rf "$DECODERS_DIR/ThreadPool"
git clone --recurse-submodules https://github.com/NVIDIA/OpenSeq2Seq
cd OpenSeq2Seq
git checkout ctc-decoders
git submodule update --init --recursive
cd "$NEMO_PATH"

mkdir -p "$DECODERS_DIR"
cp -rf OpenSeq2Seq/decoders/. "$DECODERS_DIR/"
rm -rf OpenSeq2Seq

# Verify ThreadPool landed
if [ ! -f "$DECODERS_DIR/ThreadPool/ThreadPool.h" ]; then
  echo "ERROR: ThreadPool.h still missing after clone. Aborting."
  exit 1
fi
echo "    ThreadPool.h verified."

# Copy setup.py
SETUP_SRC="$NEMO_PATH/scripts/installers/setup_os2s_decoders.py"
if [ -f "$SETUP_SRC" ]; then
  cp "$SETUP_SRC" "$DECODERS_DIR/setup.py"
  echo "    Copied setup.py"
else
  echo "ERROR: setup_os2s_decoders.py not found at $SETUP_SRC"
  exit 1
fi

# ── OpenFST ───────────────────────────────────────────────────────────────────
echo "==> Building OpenFST..."
cd "$DECODERS_DIR"

if [ ! -f "$DECODERS_DIR/openfst-1.6.3/src/lib/.libs/libfst.so" ]; then
  rm -rf "$DECODERS_DIR/openfst-1.6.3"
  wget -q \
    https://github.com/kkm000/openfst/archive/refs/tags/win/1.6.3.1.tar.gz \
    -O openfst.tar.gz
  tar -xzf openfst.tar.gz
  mv openfst-win-1.6.3.1 openfst-1.6.3
  rm -f openfst.tar.gz
  cd openfst-1.6.3
  ./configure --enable-static --enable-shared --enable-far --enable-ngram-fsts
  make -j"$(nproc)"
  cd "$DECODERS_DIR"
  echo "    OpenFST build complete."
else
  echo "    OpenFST already built, skipping."
fi

# ── KenLM ─────────────────────────────────────────────────────────────────────
echo "==> Cloning and building KenLM..."
git clone https://github.com/kpu/kenlm "$DECODERS_DIR/kenlm"
export KENLM_ROOT="$DECODERS_DIR/kenlm"

mkdir -p "$KENLM_ROOT/build"
cd "$KENLM_ROOT/build"
cmake "$KENLM_ROOT" \
  -DKENLM_MAX_ORDER=$KENLM_MAX_ORDER \
  -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"
cd "$DECODERS_DIR"
echo "    KenLM C++ build complete."

echo "==> Installing KenLM Python bindings..."
cd "$KENLM_ROOT"
python setup.py install --max_order=$KENLM_MAX_ORDER
cd "$DECODERS_DIR"

# ── ctc_decoders ──────────────────────────────────────────────────────────────
echo "==> Building ctc_decoders..."
cd "$DECODERS_DIR"

# Verify prerequisites
if [ ! -f "$KENLM_ROOT/lm/enumerate_vocab.hh" ]; then
  echo "ERROR: KenLM header missing: $KENLM_ROOT/lm/enumerate_vocab.hh"
  exit 1
fi
if [ ! -f "$DECODERS_DIR/ThreadPool/ThreadPool.h" ]; then
  echo "ERROR: ThreadPool.h missing"
  exit 1
fi

export CPLUS_INCLUDE_PATH="$KENLM_ROOT:${CPLUS_INCLUDE_PATH:-}"
python setup.py build_ext --inplace
echo "    ctc_decoders build complete."

# ── Flashlight Text ───────────────────────────────────────────────────────────
echo "==> Installing flashlight-text..."
git clone https://github.com/flashlight/text "$DECODERS_DIR/text"
cd "$DECODERS_DIR/text"
KENLM_ROOT="$KENLM_ROOT" python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall
cd "$DECODERS_DIR"

echo ""
echo "All done!"
python -c "import ctc_decoders; print('   ctc_decoders import: OK')" || echo "   ctc_decoders import: FAILED"
python -c "from flashlight.lib.text.decoder import CpuBeamSearchDecoder; print('   flashlight import: OK')" || echo "   flashlight import: FAILED"