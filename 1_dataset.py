import argparse

import numpy as np
from pyfaidx import Fasta
from tqdm import tqdm


def fasta2list(gene, chunk_size=1000000, min_len=256, max_len=1024):
    res_list = []
    for i in range(0, len(gene), chunk_size):
        chunk = gene[i:i + chunk_size]
        seq_list = chunk.split("N")

        for seq in seq_list:
            anchor = 0
            while anchor < len(seq):
                if len(seq[anchor:]) < min_len:
                    break
                elif len(seq[anchor:]) == min_len:
                    seq_len = len(seq[anchor:])
                else:
                    seq_len = np.random.randint(min_len, min(len(seq[anchor:]), max_len))

                res_list.append(seq[anchor:anchor + seq_len])
                anchor += seq_len

    return res_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fasta_file", type=str, default="./Data/NCBI/GCF_000001405.40_GRCh38.p14_genomic.fna")
    parser.add_argument("--text_file", type=str, default="./Result/TinyDNABERT-PretrainData-V1/genes.txt")

    parser.add_argument("--chunk_size", type=int, default=1000000)
    parser.add_argument("--min_len", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=1024)

    args = parser.parse_args()

    # load fasta file
    genes = Fasta(args.fasta_file, sequence_always_upper=True, as_raw=True)

    # prepare text file
    for key in tqdm(genes.keys(), desc="Processing: "):
        line_list = fasta2list(genes[key], args.chunk_size, args.min_len, args.max_len)

        with open(args.text_file, "a") as f:
            for line in line_list:
                f.write(f"{line}\n")

    # verify text file
    with open(args.text_file, "r") as f:
        lines = f.readlines()

    print(f"\nTotal: {len(lines):,}")
    print(f"Sample: {lines[0][:64]}")
