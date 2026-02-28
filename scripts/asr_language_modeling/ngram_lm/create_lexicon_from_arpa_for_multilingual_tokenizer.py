#!/usr/bin/env python

import argparse
import os
import re
import sys

from nemo.utils import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate lexicon file from a KenLM ARPA file for NeMo ASR models."
    )
    parser.add_argument(
        "--arpa", required=True, help="Path to your ARPA language model file"
    )
    parser.add_argument(
        "--dst", help="Directory to store generated lexicon", default=None
    )
    parser.add_argument(
        "--lower", action="store_true", help="Whether to lowercase ARPA vocab"
    )
    parser.add_argument(
        "--model", default=None, help="Path to NeMo .nemo model for its tokenizer"
    )
    parser.add_argument(
        "--lang_id",
        default=None,
        help="Language ID to use with multilingual tokenizer (required if model has MultilingualTokenizer)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.arpa):
        logging.critical(f"ARPA file [{args.arpa}] not found, aborting!")
        sys.exit(1)

    save_path = args.dst or os.path.dirname(args.arpa)
    os.makedirs(save_path, exist_ok=True)

    tokenizer = None
    if args.model is not None:
        from nemo.collections.asr.models import ASRModel

        model = ASRModel.restore_from(restore_path=args.model, map_location="cpu")
        if hasattr(model, "tokenizer") and model.tokenizer is not None:
            tokenizer = model.tokenizer
        else:
            logging.warning("Supplied model does not contain a tokenizer")

    lex_file = os.path.join(
        save_path, os.path.splitext(os.path.basename(args.arpa))[0] + ".lexicon"
    )
    logging.info(f"Writing Lexicon file to: {lex_file}...")

    with open(lex_file, "w", encoding="utf_8", newline="\n") as f:
        with open(args.arpa, "r", encoding="utf_8") as arpa:
            for line in arpa:
                # Only process unigram lines (format: log_prob <TAB> word <TAB> backoff)
                if not re.match(r"[-]*[0-9\.]+\t\S+\t*[-]*[0-9\.]*$", line):
                    continue
                word = line.split("\t")[1].strip()
                if args.lower:
                    word = word.lower()

                # Skip special tokens
                if word in ["<UNK>", "<unk>", "<s>", "</s>"]:
                    continue

                if tokenizer is None:
                    # Default: split letters
                    f.write(f"{word}\t{' '.join(word)}\n")
                else:
                    # Multilingual tokenizer requires lang_id
                    if "MultilingualTokenizer" in str(type(tokenizer)):
                        if args.lang_id is None:
                            raise RuntimeError(
                                "You must provide --lang_id when using a MultilingualTokenizer"
                            )
                        try:
                            tokens = tokenizer.text_to_tokens(
                                word, lang_id=args.lang_id
                            )
                        except Exception as e:
                            raise RuntimeError(
                                f"MultilingualTokenizer failed for word '{word}' with lang_id '{args.lang_id}': {e}"
                            )
                    else:
                        try:
                            tokens = tokenizer.text_to_tokens(word)
                        except Exception as e:
                            raise RuntimeError(
                                f"Tokenizer failed for word '{word}': {e}"
                            )

                    f.write(f"{word}\t{' '.join(tokens)}\n")

    logging.info("Lexicon generation complete.")
