import argparse
import os

import utils
from utils import from_json

TOKEN_CLASS = 'Fusing layers...'

TOKEN_RESULTS = 'Results saved to'
TOKEN_SPEED = 'Speed'
TOKEN_WANDB = 'Waiting for'

SW_TOP15 = [
            "bench", "chair", "bus", "bicycle", "motorcycle",
            "potted_plant", "movable_signage", "truck", "traffic_light", "traffic_sign",
            "bollard", "pole", "person", "tree_trunk", "car"
        ]

DB_TOP15 = [
    # Top 5
    "alert@Seatbelt",
    "warning@Engine",
    "alert@Parking",
    "warning@Tire",
    "warning@StabilityOn",

    # Top 10
    "alert@Brake",
    "warning@StabilityOff",
    "warning@Brake",
    "alert@Steering",
    "warning@Parking",

    # Top 15
    "alert@Retaining",
    "alert@Distance",
    "warning@ABS",
    "alert@Coolant",
    "warning@Fuel"
]

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


def tokens_in(tokens, text):
    for token in tokens:
        if token in text:
            return True
    return False


def extract_metrics(path,
                    clazzes=SW_TOP15,
                    start_tokens=[TOKEN_CLASS],
                    end_tokens=[TOKEN_RESULTS, TOKEN_SPEED, TOKEN_WANDB]):
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
                        text = text.strip('\'').strip('\n')
                        if not text:
                            continue
                        if tokens_in(end_tokens, text):
                            is_started = False
                        else:
                            tokens = text.split()
                            # TODO: need to have a better logic
                            if tokens[1].isnumeric() and tokens[2].isnumeric() and len(tokens) == 7:
                                run_table.add(tokens[0], tokens[1], tokens[2], tokens[3],
                                              tokens[4], tokens[5], tokens[6])
                    else:
                        if tokens_in(start_tokens, text):
                            is_started = True

                if len(run_table.rows) > 0:
                    runs_tables.append(run_table)

    print("Saving " + os.path.basename(path) + ".csv")
    save_by_tables(runs_tables, os.path.basename(path) + ".csv", clazzes)


def save_by_tables(runs_tables, file_out, clazzes=SW_TOP15):
    output = ""

    for run_table in runs_tables:
        row_str = ""
        for row in run_table.rows:
            row_str += "{},\n".format(row)
        output += "{}\n".format(row_str)

    # # 1. Normal top to bottom output
    # for run_table in runs_tables:
    #     output += "{}".format(run_table)
    #     # print(run_table)

    # 2. mAP5 only for all runs
    # labels = ["all"] + SW_TOP15
    labels = ["all"] + clazzes

    for label in labels:
        row_str = "{},".format(label)
        for table in runs_tables:
            value = 0
            for row in table.rows:
                if row.clazz == label:
                    value = row.map5
            row_str += "{},".format(value)

        output += "{}\n".format(row_str)

    # for id in range(max_rows):
    #     row_str = ""
    #     for run_id, run_table in enumerate(runs_tables):
    #
    #         if run_id == 0:
    #             row_str += "{},".format(run_table.rows[id])
    #         print(id, run_table.rows[0], run_table.rows[1])
    #         row_str += "{},".format(run_table.rows[id].map5)
    #
    #     output += "{}\n".format(row_str)

    utils.to_file(output, file_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_in", action="store", type=str, dest="path_in")
    parser.add_argument("--classes", action="store", type=str, dest="classes")

    args = parser.parse_args()

    clazzes = SW_TOP15
    if "dashboard15" == args.classes:
        clazzes = DB_TOP15
    print(clazzes)

    # extract_metrics('.\\notebooks\\train_sw15r_oversample_res.ipynb')
    extract_metrics(args.path_in, clazzes=clazzes)
