# Copyright (c) 2023, NVIDIA CORPORATION.
# Modified for multilingual tokenizer support (requires lang_id)

import argparse
import os
import re

from nemo.utils import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate lexicon file from KenLM ARPA file (supports multilingual tokenizers)"
    )

    parser.add_argument("--arpa", required=True, help="Path to ARPA file")
    parser.add_argument("--dst", default=None, help="Output directory for lexicon")
    parser.add_argument("--lower", action="store_true", help="Lowercase ARPA vocab")
    parser.add_argument("--model", default=None, help="Path to NeMo model (.nemo)")
    parser.add_argument(
        "--lang_id",
        default=None,
        help="Language ID (required for multilingual tokenizer models)",
    )

    args = parser.parse_args()

    # Validate ARPA
    if not os.path.exists(args.arpa):
        logging.critical(f"ARPA file [{args.arpa}] not found!")
        exit(255)

    # Output path
    save_path = args.dst if args.dst is not None else os.path.dirname(args.arpa)
    os.makedirs(save_path, exist_ok=True)

    tokenizer = None
    multilingual = False

    # Load tokenizer from model if provided
    if args.model is not None:
        from nemo.collections.asr.models import ASRModel

        logging.info(f"Loading model from: {args.model}")
        model = ASRModel.restore_from(restore_path=args.model, map_location="cpu")

        if hasattr(model, "tokenizer"):
            tokenizer = model.tokenizer
            logging.info(f"Tokenizer type: {type(tokenizer)}")

            # Detect multilingual tokenizer
            if "Multilingual" in tokenizer.__class__.__name__:
                multilingual = True
                logging.info("Detected MultilingualTokenizer")

                if args.lang_id is None:
                    logging.critical(
                        "This model uses MultilingualTokenizer. Please provide --lang_id"
                    )
                    exit(255)
        else:
            logging.warning("Model does not contain a tokenizer")

    # Output lexicon file
    lex_file = os.path.join(
        save_path,
        os.path.splitext(os.path.basename(args.arpa))[0] + ".lexicon",
    )

    logging.info(f"Writing Lexicon file to: {lex_file}")

    with open(lex_file, "w", encoding="utf_8", newline="\n") as fout:
        with open(args.arpa, "r", encoding="utf_8") as arpa:
            for line in arpa:
                # Match unigram lines only
                if not re.match(r"[-]*[0-9\.]+\t\S+\t*[-]*[0-9\.]*$", line):
                    continue

                word = line.split("\t")[1]
                word = word.strip().lower() if args.lower else word.strip()

                # Skip special tokens
                if word in ["<UNK>", "<unk>", "<s>", "</s>"]:
                    continue

                # No tokenizer → character lexicon fallback
                if tokenizer is None:
                    fout.write(f"{word}\t{' '.join(word)}\n")
                    continue

                try:
                    # Multilingual tokenizer requires lang_id
                    if multilingual:
                        w_ids = tokenizer.text_to_ids(word, lang_id=args.lang_id)
                        tokens = tokenizer.text_to_tokens(word, lang_id=args.lang_id)
                    else:
                        w_ids = tokenizer.text_to_ids(word)
                        tokens = tokenizer.text_to_tokens(word)

                    # Skip words that map to <unk>
                    if tokenizer.unk_id not in w_ids:
                        fout.write(f"{word}\t{' '.join(tokens)}\n")

                except Exception as e:
                    logging.warning(f"Skipping word '{word}' due to error: {e}")

    logging.info("Lexicon generation complete.")
