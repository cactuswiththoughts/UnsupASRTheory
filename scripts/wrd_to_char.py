#!/usr/bin/env python3 -u

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inpath")
    parser.add_argument("outpath")
    args = parser.parse_args()

    with open(args.inpath, "r") as fin,\
        open(args.outpath, "w") as fout:
        for line in fin:
            chars = [c for w in line.split() for c in w] 
            fout.write(" ".join(chars)+"\n")

if __name__ == "__main__":
    main()
