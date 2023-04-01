import os
import csv
import glob


def merge_csv_files(path_annts, sp):
    """
    Merge multiple csv files into a single file
    """

    csv_files = sorted(glob.glob(os.path.join(path_annts, '*.csv')))
    data = []
    for csv_file in csv_files:
        with open(csv_file) as f:
            reader = csv.reader(f)
            data.extend(list(reader))

    with open(f'data/annotations/ava_activespeaker_{sp}_v1.0.csv', 'w') as f:
        writer = csv.writer(f, delimiter =',')
        writer.writerows(data)


if __name__ == "__main__":
    path_annts = 'data/annotations/ava_activespeaker_test_v1.0'
    sp = 'val'

    merge_csv_files(path_annts, sp)
