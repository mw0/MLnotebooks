#!/usr/bin/python3

inFileName = './USCensusCountyAdjacencyTable.tsv'
outFileName = './CountyAdjacencyTable.tsv'

with open(outFileName, 'w', encoding='utf-8') as outFile:
    outFile.write('County\tFIPS\tAdjacentFIPS\n')
    with open(inFileName, 'r', encoding='ISO-8859-1') as inFile:
        all = inFile.readlines()

        _ = all.pop(0)		# Remove header


        for i, line in enumerate(all):
            # if i == 40:
            #     break
            elements = line.strip().split('\t')

            if i == 0:
                AdjacentFIPSs = []
                lastCounty, lastCountyFIPS, _, _ = elements
                lastCounty = lastCounty.strip('"')
                print(elements)
            if i > 0:
                if len(elements) == 2:
                    print(elements)
                    AdjacentFIPSs.append(elements[1])
                elif len(elements) == 4:
                    County, CountyFIPS, AdjacentCounty, AdjacentFIPS = elements
                    AdjacentFIPSs = list(set(AdjacentFIPSs))
                    AdjacentFIPSs = list(filter(lastCountyFIPS.__ne__,
                                                AdjacentFIPSs))
                    outFile.write('\t'.join([lastCounty, lastCountyFIPS,
                                             ",".join(sorted(AdjacentFIPSs))])
                                  + "\n")
                    print('\t'.join([lastCounty, lastCountyFIPS,
                                     ",".join(sorted(AdjacentFIPSs))]))
                    print(elements)
                    AdjacentFIPSs = [AdjacentFIPS]
                    lastCounty = County
                    lastCountyFIPS = CountyFIPS

        print(elements)
        if len(elements) == 2:
            AdjacentFIPSs = list(set(AdjacentFIPSs))
            AdjacentFIPSs = list(filter(lastCountyFIPS.__ne__, AdjacentFIPSs))
            outFile.write('\t'.join([lastCounty, lastCountyFIPS,
                                     ",".join(sorted(AdjacentFIPSs))]) + "\n")
            print('\t'.join([lastCounty, lastCountyFIPS,
                             ",".join(sorted(AdjacentFIPSs))]))
