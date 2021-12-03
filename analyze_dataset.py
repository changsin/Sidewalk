import argparse
import os

import numpy as np
import pandas as pd
from lxml import etree

import utils
from utils import glob_files, glob_folders

DATA_ROOT = 'E:\\Sidewalk\\Sidewalk_19009_output_zip\\Sidewalk_19009_output_zip_box\\Sidewalk_19009_output_zip_box_000001_1\\archive\\sw15_train_r'
SIDEWALK_CLASSES = ['barricade', 'bench', 'bicycle', 'bollard', 'bus',
                    'car', 'carrier', 'cat', 'chair', 'dog',
                    'fire_hydrant', 'kiosk', 'motorcycle', 'movable_signage', 'parking_meter',
                    'person', 'pole', 'potted_plant', 'power_controller', 'scooter',
                    'stop', 'stroller', 'table', 'traffic_light', 'traffic_light_controller',
                    'traffic_sign', 'tree_trunk', 'truck', 'wheelchair', ]

SW_TOP15 = ["bench", "chair", "bus", "bicycle", "motorcycle",
            "potted_plant", "movable_signage", "truck", "traffic_light", "traffic_sign",
            "bollard", "pole", "person", "tree_trunk", "car"]


def parse_xml(filename):
    image_labels = []

    tree = etree.parse(filename)
    for image in tree.xpath('image'):
        # print(image.attrib['name'])
        name = image.attrib['name']
        width = int(image.attrib['width'])
        height = int(image.attrib['height'])

        boxes = []

        for box in image.xpath('box'):
            xtl = float(box.attrib['xtl'])
            ytl = float(box.attrib['ytl'])
            xbr = float(box.attrib['xbr'])
            ybr = float(box.attrib['ybr'])

            label = box.attrib['label']
            # wtype = box.xpath('attribute[@name="name"]')[0].text
            # daynight = box.xpath('attribute[@name="daynight"]')[0].text
            # visibility = int(box.xpath('attribute[@name="visibility"]')[0].text)

            # box = wtype, alertwarning, daynight, visibility, xtl, ytl, xbr, ybr
            box = label, xtl, ytl, xbr, ybr

            boxes.append(box)

        image_labels.append([name, width, height, np.array(boxes)])

    return np.array(image_labels)


def load_labels(path, file_type='*'):
    files = glob_files(path, file_type=file_type)

    if files is None:
        folders = glob_folders(path, file_type='*')
        for folder in folders:
            files.extend(glob_files(folder, file_type=file_type))
    print(files)

    y = []
    dfy = []

    for file in files:
        print(f"Parsing {file}")
        labels = parse_xml(file)
        y.append([os.path.basename(file), labels])
        for label in labels:
            filename = label[0]
            width = label[1]
            height = label[2]
            boxes = label[3]
            for box in boxes:
                # wtype = box[0]
                # alertwarning = box[1]
                # day = box[2]
                # visibility = box[3]
                label = box[0]

                xtl = box[1]
                ytl = box[2]
                xbr = box[3]
                ybr = box[4]

                dfy.append([os.path.basename(file), filename, width, height, label, xtl, ytl, xbr, ybr])

    return np.array(y), np.array(dfy)


def count_labels_per_folder(dfyy):
    def _count_labels_per_folder():
        dfy = pd.DataFrame.from_records(dfyy)
        dfy.head()

        dfy.columns = ['folder', 'filename', 'width', 'height', 'class', 'xtl', 'ytl', 'xbr', 'ybr']
        dfy.head()

        dfy.drop(['width', 'height', 'xtl', 'ytl', 'xbr', 'ybr'], inplace=True, axis=1)
        dfy.head()

        folder_label_counts = []
        folder_names = dfy['folder'].unique()

        for folder_name in folder_names:
            df_folder = dfy.loc[dfy['folder'] == folder_name]
            label_counts = df_folder['class'].value_counts().to_dict()
            folder_label_counts.append((folder_name, label_counts))
            # folder_label_counts[folder_name] = label_counts

        return folder_label_counts

    label_counts_per_folder = _count_labels_per_folder()
    rows = ""

    row = "{},".format("class")
    for folder_name, _ in label_counts_per_folder:
        row += "{},".format(folder_name[10:-4])
    rows += "{}\n".format(row)

    for clazz in SW_TOP15:
        row = "{},".format(clazz)
        for _, label_counts in label_counts_per_folder:
            value = 0
            if clazz in label_counts:
                value = label_counts[clazz]
            row += "{},".format(value)
        rows += "{}\n".format(row)

        # class_counts.append((clazz, classes.count(clazz)))
        # class_counts.append([clazz, int(classes.count(clazz))])
        # print(f'{clazz}: {classes.count(clazz)}')
        # print(f'\'{clazz}\', ', end='')

    # sort tuples array
    # class_counts.sort(key=lambda x: x[1])

    # class_counts = class_counts[np.argsort(class_counts[:, 1])]

    utils.to_file(rows, 'label_counts.csv')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", action="store", dest="mode")

    y, dfyy = load_labels(DATA_ROOT, file_type='*.xml')

    count_labels_per_folder(dfyy)
