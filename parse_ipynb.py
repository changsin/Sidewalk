import argparse

import utils
from utils import from_json

TOKEN_CLASS = 'Class'
TOKEN_SPEED = 'Speed'


class RunTable:
    class Row:
        def __init__(self, clazz, image_count, label_count, P, R, map5, map95):
            self.clazz = clazz
            self.image_count = int(image_count)
            self.label_count = int(label_count)
            self.P = float(P)
            self.R = float(R)
            self.map5 = float(map5)
            self.map95 = float(map95)

        def __str__(self):
            return "{},{},{},{},{},{},{}".format(self.clazz, self.image_count, self.label_count,
                                                 self.P, self.R, self.map5, self.map95)

    def __init__(self, name):
        self.name = name
        self.rows = []

    def add(self, clazz, image_count, label_count, P, R, map5, map95):
        self.rows.append(RunTable.Row(clazz, image_count, label_count, P, R, map5, map95))

    def __str__(self):
        table = "{}\n".format(self.name)
        for row in self.rows:
            table += "{}\n".format(row)
        return table


def extract_metrics(path):

    json_obj = from_json(path)
    cells = json_obj['cells']

    runs_tables = []

    is_started = False
    for cell in cells:
        if cell['cell_type'] == 'code':
            outputs = cell['outputs']
            if outputs:
                run_table = RunTable(cell['source'][0])

                texts = outputs[0]['text']
                for text in texts:
                    if is_started:
                        if TOKEN_SPEED in text:
                            is_started = False
                        else:
                            text = text.strip('\'').strip("\\n")
                            tokens = text.split()
                            run_table.add(tokens[0], tokens[1], tokens[2], tokens[3],
                                          tokens[4], tokens[5], tokens[6])
                    else:
                        if TOKEN_CLASS in text:
                            is_started = True

                if len(run_table.rows) > 0:
                    runs_tables.append(run_table)

    output = ""
    # # 1. Normal top to bottom output
    # for run_table in runs_tables:
    #     output += "{}".format(run_table)
    #     # print(run_table)

    # 2. mAP5 only for all runs
    num_rows = len(runs_tables[0].rows)

    for id in range(num_rows):
        row = ""
        for run_id, run_table in enumerate(runs_tables):
            if run_id == 0:
                row += "{},".format(run_table.rows[id])
            row += "{},".format(run_table.rows[id].map5)

        output += "{}\n".format(row)

    utils.to_file(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", action="store", dest="mode")

    extract_metrics('.\\notebooks\\train_sw15_oversample_res.ipynb')
