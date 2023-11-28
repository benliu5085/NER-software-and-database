import os
from utils.bmes_decode import bmes_decode
import json
import sys


def convert_file(input_file, output_file, tag2query_file):
    origin_count = 0
    new_count = 0
    tag2query = json.load(open(tag2query_file))
    mrc_samples = []
    with open(input_file) as fin:
        for (cnt_line, line) in enumerate(fin):
            line = line.strip()
            print(line)
            if not line:
                continue
            origin_count += 1
            src, labels, _ = line.split("\t")
            print(src)
            print(labels)
            tags = bmes_decode(char_label_list=[(char, label) for char, label in zip(src.split(), labels.split())])
            for label, query in tag2query.items():
                mrc_samples.append(
                    {
                        "context": src,
                        "start_position": [tag.begin for tag in tags if tag.tag == label],
                        "end_position": [tag.end-1 for tag in tags if tag.tag == label],
                        "query": query,
                        "qas_id": "{}.{}".format(cnt_line, list(tag2query).index(label)+1),
                        "entity_label": label,
                        "impossible": len([tag.begin for tag in tags if tag.tag == label]) == 0,
                        "span_position": ["{};{}".format(tag.begin, tag.end-1) for tag in tags if tag.tag == label],
                    }
                )
                new_count += 1

    json.dump(mrc_samples, open(output_file, "w"), ensure_ascii=False, sort_keys=True, indent=2)
    print(f"Convert {origin_count} samples to {new_count} samples and save to {output_file}")


def main():
    if len(sys.argv) == 2:
        fdir = sys.argv[1]
    else:
        print("call by ~ <fdir>")
        print("     <fdir> should be the name one of directory in ~/ner2mrc")
        exit()

    raw_dir = f"ner2mrc/{fdir}"
    mrc_dir = f"datasets/{fdir}"
    tag2query_file = f"ner2mrc/queries/bio.json"
    os.makedirs(mrc_dir, exist_ok=True)
    fout = open(f"{mrc_dir}/example.txt", 'w')
    for phase in os.listdir(raw_dir):
        phase = phase[:-4]
        fout.write(f"{phase}\n")
        old_file = os.path.join(raw_dir, f"{phase}.tsv")
        new_file = os.path.join(mrc_dir, f"mrc-ner.{phase}")
        convert_file(old_file, new_file, tag2query_file)

    fout.close()

if __name__ == '__main__':
    main()
