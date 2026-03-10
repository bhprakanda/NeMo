# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#
# Use this file to create a lexicon file for Flashlight decoding from an existing KenLM arpa file
# A lexicon file is required for Flashlight decoding in most cases, as it acts as a map from the words
# in your arpa file to the representation used by your ASR AM.
# For more details, see: https://github.com/flashlight/flashlight/tree/main/flashlight/app/asr#data-preparation
#
# Usage (normal model):
#   python create_lexicon_from_arpa.py --arpa /path/to/english.arpa --model /path/to/model.nemo --lower
#
# Usage (multilingual model):
#   python create_lexicon_from_arpa.py --arpa /path/to/nepali.arpa --model /path/to/model.nemo --lang_id ne
#   python create_lexicon_from_arpa.py --arpa /path/to/maithili.arpa --model /path/to/model.nemo --lang_id mai


import argparse
import os
import re

from nemo.utils import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility script for generating lexicon file from a KenLM arpa file"
    )
    parser.add_argument("--arpa", required=True, help="path to your arpa file")
    parser.add_argument(
        "--dst", help="directory to store generated lexicon", default=None
    )
    parser.add_argument(
        "--lower", action="store_true", help="Whether to lowercase the arpa vocab"
    )
    parser.add_argument(
        "--model", default=None, help="path to Nemo model for its tokenizer"
    )
    parser.add_argument(
        "--lang_id",
        default=None,
        help="Language ID for multilingual models (e.g. 'ne' for Nepali, 'mai' for Maithili). "
        "Required when using a model with a MultilingualTokenizer.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.arpa):
        logging.critical(f"ARPA file [ {args.arpa} ] not detected on disk, aborting!")
        exit(255)

    if args.dst is not None:
        save_path = args.dst
    else:
        save_path = os.path.dirname(args.arpa)
    os.makedirs(save_path, exist_ok=True)

    tokenizer = None
    lang_tokenizer = None  # per-language tokenizer for multilingual models
    is_multilingual = False

    if args.model is not None:
        from nemo.collections.asr.models import ASRModel
        from nemo.collections.common.tokenizers.multilingual_tokenizer import (
            MultilingualTokenizer,
        )

        model = ASRModel.restore_from(restore_path=args.model, map_location="cpu")

        if hasattr(model, "tokenizer"):
            tokenizer = model.tokenizer

            # Check if this is a multilingual tokenizer
            if isinstance(tokenizer, MultilingualTokenizer):
                is_multilingual = True

                if args.lang_id is None:
                    logging.critical(
                        "Model has a MultilingualTokenizer but --lang_id was not provided. "
                        f"Please specify one of: {list(tokenizer.tokenizers_dict.keys())}"
                    )
                    exit(255)

                if args.lang_id not in tokenizer.tokenizers_dict:
                    logging.critical(
                        f"lang_id '{args.lang_id}' not found in model. "
                        f"Available languages: {list(tokenizer.tokenizers_dict.keys())}"
                    )
                    exit(255)

                lang_tokenizer = tokenizer.tokenizers_dict[args.lang_id]
                vocab_set = set(lang_tokenizer.vocab)
                logging.info(
                    f"Multilingual model detected. Using tokenizer for lang_id='{args.lang_id}' "
                    f"with vocab size {len(vocab_set)}."
                )
            else:
                logging.info("Single-language tokenizer detected.")
        else:
            logging.warning("Supplied NeMo model does not contain a tokenizer")

    lex_file = os.path.join(
        save_path, os.path.splitext(os.path.basename(args.arpa))[0] + ".lexicon"
    )

    logging.info(f"Writing Lexicon file to: {lex_file}...")

    written, skipped = 0, 0

    with open(lex_file, "w", encoding="utf_8", newline="\n") as f:
        with open(args.arpa, "r", encoding="utf_8") as arpa:
            for line in arpa:
                # verify if the line corresponds to unigram
                if not re.match(r"[-]*[0-9\.]+\t\S+\t*[-]*[0-9\.]*$", line):
                    continue
                word = line.split("\t")[1]
                word = word.strip().lower() if args.lower else word.strip()
                if word in ("<UNK>", "<unk>", "<s>", "</s>"):
                    continue

                if tokenizer is None:
                    # No model provided - use character-level lexicon
                    f.write("{w}\t{s}\n".format(w=word, s=" ".join(word)))
                    written += 1

                elif is_multilingual:
                    # Multilingual tokenizer - use per-language tokenizer directly
                    try:
                        tokens = lang_tokenizer.text_to_tokens(word)
                        if not tokens:
                            skipped += 1
                            continue
                        # Skip if any token is unknown
                        if any(t not in vocab_set for t in tokens):
                            skipped += 1
                            continue
                        f.write("{w}\t{s}\n".format(w=word, s=" ".join(tokens)))
                        written += 1
                    except Exception as e:
                        logging.debug(f"Skipping word '{word}': {e}")
                        skipped += 1

                else:
                    # Normal single-language tokenizer - original behaviour
                    w_ids = tokenizer.text_to_ids(word)
                    if tokenizer.unk_id not in w_ids:
                        f.write(
                            "{w}\t{s}\n".format(
                                w=word, s=" ".join(tokenizer.text_to_tokens(word))
                            )
                        )
                        written += 1
                    else:
                        skipped += 1

    logging.info(f"Done. Written: {written} words, Skipped: {skipped} words.")
