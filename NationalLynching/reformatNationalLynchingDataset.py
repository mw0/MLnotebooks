#!/usr/bin/python3

inFileName = './NationalLynchingDataset1883-1941.Seguin.C.etal.2019.csv'
outFileName = './NationalLynchingDataset1883-1941.Seguin.C.etal.2019.tsv'

import csv

with open(outFileName, 'w', encoding='utf-8') as outFile:
    with open(inFileName, 'r', encoding='ISO-8859-1') as inFile:
        all = inFile.readlines()

        header = (all.pop(0).strip() + 's').split(',')
        print(header)
        outFile.write("\t".join(header) + "\n")

        lines = csv.reader(all, skipinitialspace=True)

        for i, line in enumerate(lines):
            if i < 30:
                print(f"{i:03}, {len(line):02}")
                print("\t".join(line))
            outFile.write("\t".join(line) + "\n")
