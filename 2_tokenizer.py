import argparse
import os
import tempfile

from tokenizers.implementations import CharBPETokenizer
from transformers import RobertaTokenizer


def subsample(original_file, sample_rate):
    temp = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt")
    with open(original_file, "r", encoding="utf-8") as f_in:
        for i, line in enumerate(f_in):
            if i % sample_rate == 0:
                temp.write(line)
    temp.close()

    return temp.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--text_file", type=str, default="./Result/TinyDNABERT-PretrainData-V1/genes.txt")
    parser.add_argument("--init_path", type=str, default="./Result/TinyDNABERT-20M-V1-Init/")
    parser.add_argument("--final_path", type=str, default="./Result/TinyDNABERT-20M-V1-Final/")

    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--min_frequency", type=int, default=4)
    parser.add_argument("--sample_rate", type=int, default=10)

    args = parser.parse_args()

    # subsample text file
    sampled_file = subsample(args.text_file, sample_rate=args.sample_rate)

    # train BPE tokenizer
    tokenizer = CharBPETokenizer()
    tokenizer.train(
        files=[sampled_file],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"],
        show_progress=True,
    )

    # save BPE tokenizer
    os.remove(sampled_file)
    tokenizer.save_model(args.init_path)
    # tokenizer.save_model(args.final_path)

    tokenizer = RobertaTokenizer.from_pretrained(args.init_path)
    tokenizer.save_pretrained(args.init_path)
    tokenizer.save_pretrained(args.final_path)

    # verify BPE tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.final_path)
    print("Vocab Size:", tokenizer.vocab_size)

    gene = "CAGGCGCAGAGACACATGCTACCGCGTCCAGGGGTGGAGGCGTGGCGCAGGCGCAGAGAGGCGCACCG"
    print("\nOriginal:", gene)
    print("Tokenized:", tokenizer.tokenize(gene))
    print("Encoded:", tokenizer.encode(gene))
