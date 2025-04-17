import argparse
import csv


def main(args):
    prefix = '.'.join(args.file.split('.')[:-1])

    with open(args.file) as fp, \
         open(prefix + '.src', 'w') as ofp1, \
         open(prefix + '.tgt', 'w') as ofp2:
        reader = csv.reader(fp)
        para = 0
        mono = 0
        for i, row in enumerate(reader):
            if i == 0:
                k = row
            else:
                d = {k[j]: row[j] for j in range(len(row))}
                ofp1.write(repr(d['context']).strip("'").strip('"').strip() + '\n')
                ofp2.write(repr(d['response']).strip("'").strip('"').strip() + '\n')
                para += 1
        print(para, mono)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file')

    args = parser.parse_args()
    main(args)