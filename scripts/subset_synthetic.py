import argparse
import numpy as np
import shutil
from pathlib import Path
np.random.seed(2)


def main(
    in_path, 
    sp_subset_size,
    txt_subset_size, 
    out_path,
):
    in_path = Path(in_path)
    out_path = Path(out_path)

    # Create a new directory
    if not out_path.exists():
        out_path.mkdir(
            parents=True, exist_ok=True
        )
    
    # Copy unchanged files
    for suffix in ["tsv", "npy", "lengths", "wrd", "phn"]:
        shutil.copyfile(
            in_path / f"valid.{suffix}", 
            out_path / f"valid.{suffix}",
        )
     
    # Subset the .tsv file
    with open(in_path / "train.tsv", "r") as f_in,\
        open(out_path / "train.tsv", "w") as f_out,\
        open(in_path / "valid.tsv", "r") as f_valid,\
        open(out_path / "all.tsv", "w") as f_all:
        in_lines = f_in.read().strip().split("\n")
        n_examples = len(in_lines) - 1
        sp_subset_idxs = np.random.permutation(
            n_examples
        )[:sp_subset_size]
        txt_subset_idxs = np.random.permutation(
            n_examples
        )[:txt_subset_size]

        out_lines = [in_lines[0]] + \
            [in_lines[i+1] for i in sp_subset_idxs]
        f_out.write("\n".join(out_lines))
        
        valid_lines = f_valid.read().strip().split("\n")
        f_all.write(
            "\n".join(out_lines+valid_lines[1:])
        )

    # Subset the feature and .lengths file
    with open(in_path / "train.lengths", "r") as f_in,\
        open(out_path / "train.lengths", "w") as f_out,\
        open(in_path / "valid.lengths", "r") as f_valid,\
        open(out_path / "all.lengths", "w") as f_all:
        in_lens = f_in.read().strip().split("\n")
        out_lens = [in_lens[i] for i in sp_subset_idxs]
        f_out.write("\n".join(out_lens))
        valid_lens = f_valid.read().strip().split("\n")
        f_all.write(
            "\n".join(out_lens+valid_lens)
        )
        in_feat = np.load(in_path / "train.npy")
        out_feat = np.concatenate(
            [
                in_feat[i*int(l):(i+1)*int(l)] 
                for i, l in zip(sp_subset_idxs, out_lens)
            ]
        )
        valid_feat = np.load(in_path / "valid.npy")
        all_feat = np.concatenate([out_feat, valid_feat])
        np.save(out_path / "train.npy", out_feat)
        np.save(out_path / "all.npy", all_feat)

    # Subset the .wrd and .phn file
    with open(in_path / "train.wrd", "r") as f_in,\
        open(out_path / "train.wrd", "w") as f_out,\
        open(in_path / "valid.wrd", "r") as f_valid,\
        open(out_path / "all.wrd", "w") as f_all:
        in_sents = f_in.read().strip().split("\n")
        out_sents = [in_sents[i] for i in txt_subset_idxs]
        f_out.write("\n".join(out_sents))
        valid_sents = f_valid.read().strip().split("\n")
        f_all.write(
            "\n".join(out_sents+valid_sents)
        )

    with open(in_path / "train.phn", "r") as f_in,\
        open(out_path / "train.phn", "w") as f_out,\
        open(in_path / "valid.phn", "r") as f_valid,\
        open(out_path / "all.phn", "w") as f_all:
        in_sents = f_in.read().strip().split("\n")
        out_sents = [in_sents[i] for i in txt_subset_idxs]
        f_out.write("\n".join(out_sents))
        all_sents = f_valid.read().strip().split("\n")
        f_all.write(
            "\n".join(out_sents+valid_sents)
        )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path")
    parser.add_argument("--speech_subset_size", type=int)
    parser.add_argument("--text_subset_size", type=int)
    parser.add_argument("--out_path")
    args = parser.parse_args()
    main(
        args.in_path,
        args.speech_subset_size,
        args.text_subset_size,
        args.out_path,
    )
