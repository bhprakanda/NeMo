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

# Use this script to install KenLM, OpenSeq2Seq decoder, Flashlight decoder
shopt -s expand_aliases

NEMO_PATH=/workspace/nemo  # Path to NeMo folder: /workspace/nemo if you use NeMo/Dockerfile
if [ "$#" -eq 1 ]; then
  NEMO_PATH=$1
fi
KENLM_MAX_ORDER=10 # Maximum order of KenLM model, also specified in the setup_os2s_decoders.py

if [ -d "$NEMO_PATH" ]; then
  echo "The folder '$NEMO_PATH' exists."
else
  echo "Error: The folder '$NEMO_PATH' does not exist. Specify it as a first command line positional argument!"
  exit 1
fi
cd $NEMO_PATH

if [ $(id -u) -eq 0 ]; then
  alias aptupdate='apt-get update'
else
  alias aptupdate='sudo apt-get update'
fi

aptupdate && apt-get upgrade -y && apt-get install -y swig liblzma-dev libboost-all-dev && rm -rf /var/lib/apt/lists/*

# Use system Boost installed via apt (replaces broken JFrog download URL)
export BOOST_ROOT=/usr

git clone https://github.com/NVIDIA/OpenSeq2Seq
cd OpenSeq2Seq
git checkout ctc-decoders
cd ..
mv OpenSeq2Seq/decoders $NEMO_PATH/
rm -rf OpenSeq2Seq
cd $NEMO_PATH/decoders
cp $NEMO_PATH/scripts/installers/setup_os2s_decoders.py ./setup.py

# Download OpenFST from GitHub mirror (replaces dead openfst.org URL)
wget https://github.com/kkm000/openfst/archive/refs/tags/win/1.6.3.1.tar.gz -O openfst.tar.gz
tar -xzf openfst.tar.gz
mv openfst-win-1.6.3.1 openfst-1.6.3
cd openfst-1.6.3
./configure --enable-static --enable-shared --enable-far --enable-ngram-fsts
make -j4
cd ..

python setup.py build_ext --inplace

# install KenLM
cd $NEMO_PATH/decoders/kenlm/build && cmake -DKENLM_MAX_ORDER=$KENLM_MAX_ORDER .. && make -j2
cd $NEMO_PATH/decoders/kenlm
python setup.py install --max_order=$KENLM_MAX_ORDER
export KENLM_LIB=$NEMO_PATH/decoders/kenlm/build/bin
export KENLM_ROOT=$NEMO_PATH/decoders/kenlm
cd ..

# install Flashlight
git clone https://github.com/flashlight/text && cd text
python setup.py bdist_wheel
pip install dist/*.whl
cd ..