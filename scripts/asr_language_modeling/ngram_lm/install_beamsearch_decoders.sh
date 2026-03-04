#!/usr/bin/env bash
set -euo pipefail

NEMO_PATH=/workspace/nemo
if [ "$#" -eq 1 ]; then
  NEMO_PATH=$1
fi
KENLM_MAX_ORDER=10

if [ ! -d "$NEMO_PATH" ]; then
  echo "Error: '$NEMO_PATH' does not exist. Pass it as the first argument."
  exit 1
fi
echo "==> Using NEMO_PATH=$NEMO_PATH"

# ── System dependencies ───────────────────────────────────────────────────────
echo "==> Installing system dependencies..."
APT="apt-get"
[ "$(id -u)" -ne 0 ] && APT="sudo apt-get"
$APT update -qq
$APT install -y -qq swig liblzma-dev libboost-all-dev cmake git wget build-essential
export BOOST_ROOT=/usr

# ── OpenSeq2Seq decoders ──────────────────────────────────────────────────────
echo "==> Setting up OpenSeq2Seq decoders..."
cd "$NEMO_PATH"

DECODERS_DIR="$NEMO_PATH/decoders"

if [ ! -f "$DECODERS_DIR/setup.py" ]; then
  rm -rf OpenSeq2Seq
  git clone https://github.com/NVIDIA/OpenSeq2Seq
  cd OpenSeq2Seq
  git checkout ctc-decoders
  cd ..

  # mkdir -p handles the case where decoders/ does not exist yet
  # cp -rn handles the case where it already exists (no-clobber merge)
  mkdir -p "$DECODERS_DIR"
  cp -rn OpenSeq2Seq/decoders/. "$DECODERS_DIR/"
  rm -rf OpenSeq2Seq

  SETUP_SRC="$NEMO_PATH/scripts/installers/setup_os2s_decoders.py"
  if [ -f "$SETUP_SRC" ]; then
    cp "$SETUP_SRC" "$DECODERS_DIR/setup.py"
  else
    echo "Warning: setup_os2s_decoders.py not found at $SETUP_SRC"
  fi
else
  echo "    setup.py already present, skipping OpenSeq2Seq clone."
fi

cd "$DECODERS_DIR"

# ── OpenFST ───────────────────────────────────────────────────────────────────
echo "==> Building OpenFST..."
if [ ! -f "openfst-1.6.3/src/lib/.libs/libfst.so" ]; then
  rm -rf openfst-1.6.3 openfst.tar.gz
  wget -q https://github.com/kkm000/openfst/archive/refs/tags/win/1.6.3.1.tar.gz -O openfst.tar.gz
  tar -xzf openfst.tar.gz
  mv openfst-win-1.6.3.1 openfst-1.6.3
  rm -f openfst.tar.gz
  cd openfst-1.6.3
  ./configure --enable-static --enable-shared --enable-far --enable-ngram-fsts
  make -j"$(nproc)"
  cd ..
else
  echo "    OpenFST already built, skipping."
fi

# ── KenLM ─────────────────────────────────────────────────────────────────────
echo "==> Setting up KenLM..."
if [ ! -f "kenlm/CMakeLists.txt" ]; then
  rm -rf kenlm
  git clone https://github.com/kpu/kenlm kenlm
else
  echo "    kenlm source already present, skipping clone."
fi

export KENLM_ROOT="$DECODERS_DIR/kenlm"

if [ ! -f "$KENLM_ROOT/build/lib/libkenlm.a" ]; then
  mkdir -p "$KENLM_ROOT/build"
  cd "$KENLM_ROOT/build"
  cmake .. -DKENLM_MAX_ORDER=$KENLM_MAX_ORDER -DCMAKE_BUILD_TYPE=Release
  make -j"$(nproc)"
  cd "$DECODERS_DIR"
else
  echo "    KenLM already built, skipping cmake/make."
fi

echo "==> Installing KenLM Python bindings..."
if ! python -c "import kenlm" 2>/dev/null; then
  cd "$KENLM_ROOT"
  python setup.py install --max_order=$KENLM_MAX_ORDER
  cd "$DECODERS_DIR"
else
  echo "    kenlm Python package already installed, skipping."
fi

# ── ctc_decoders ──────────────────────────────────────────────────────────────
echo "==> Building ctc_decoders..."

if [ ! -f "$DECODERS_DIR/setup.py" ]; then
  echo "Error: setup.py missing from $DECODERS_DIR. Cannot build ctc_decoders."
  exit 1
fi

# Expose all KenLM header paths the compiler needs
export CPLUS_INCLUDE_PATH="$KENLM_ROOT:$KENLM_ROOT/lm:${CPLUS_INCLUDE_PATH:-}"

cd "$DECODERS_DIR"
python setup.py build_ext --inplace

# ── Flashlight Text ───────────────────────────────────────────────────────────
echo "==> Installing flashlight-text..."
if ! python -c "from flashlight.lib.text.decoder import CpuBeamSearchDecoder" 2>/dev/null; then
  cd "$DECODERS_DIR"
  rm -rf text
  git clone https://github.com/flashlight/text
  cd text
  KENLM_ROOT="$KENLM_ROOT" python setup.py bdist_wheel
  pip install dist/*.whl --force-reinstall
  cd "$DECODERS_DIR"
else
  echo "    flashlight-text already installed, skipping."
fi

echo ""
echo "Done! ctc_decoders and flashlight-text are installed."