import argparse
import os
import re
from pathlib import Path

import cv2
import pandas as pd
import tqdm

ROOT_DIR = os.path.dirname(Path(__file__).parent.parent.absolute())
DATA_DIR = os.path.join(ROOT_DIR, "data")


def crop_and_write_cells(src_path, label, img_list, df_cells):
    for img in tqdm.tqdm(img_list):
        try:
            img_name = os.path.splitext(img)[0]
            df_cell = df_cells[df_cells["img_name"] == img].reset_index()
            img = cv2.imread(f"{src_path}/{img}")
            for cell_idx in range(df_cell.shape[0]):
                coords = re.findall(r"\d+", df_cell.iloc[cell_idx, :]["bbox"])
                cell_type = df_cell.loc[cell_idx, "class"]
                coords = list(map(int, coords))
                cropped_cell = img[coords[1] : coords[3], coords[0] : coords[2], :]
                os.makedirs(Path(DATA_DIR) / label, exist_ok=True)
                cv2.imwrite(
                    f"{DATA_DIR}/{label}/{img_name}_{cell_type}_{cell_idx}.jpg",
                    cropped_cell,
                )
        except Exception as e:
            print(img_name, e)


def main():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [SRC_PATH]...",
        description="Path of the source directory of the data",
    )
    parser.add_argument("-p", "--path", type=str, required=True)
    args = parser.parse_args()
    src_path = args.path
    df_cells = pd.read_csv(os.path.join(src_path, "cells.csv"))
    df_cells = df_cells.fillna("nan")
    df_cells_count = df_cells.groupby(["img_name", "class"]).size().unstack()
    df_cells_count = df_cells_count.fillna(0).reset_index()
    train_valid_df = df_cells_count.groupby(
        ["RBC", "Platelets", "WBC", "nan"], group_keys=False
    ).apply(lambda x: x.sample(frac=0.6))
    valid_images = list(
        train_valid_df.groupby(
            ["RBC", "Platelets", "WBC", "nan"], group_keys=False
        ).apply(lambda x: x.sample(frac=0.3))["img_name"]
    )
    train_images = list(
        train_valid_df[~train_valid_df["img_name"].isin(valid_images)]["img_name"]
    )
    test_images = list(
        df_cells_count[~df_cells_count["img_name"].isin(train_valid_df["img_name"])][
            "img_name"
        ]
    )
    assert set(valid_images).intersection(set(train_images)) == set()
    assert set(valid_images).intersection(set(test_images)) == set()
    assert set(valid_images).intersection(set(test_images)) == set()

    data_dict = {
        "train_cells": train_images,
        "valid_cells": valid_images,
        "test_cells": test_images,
    }
    for label, img_list in data_dict.items():
        crop_and_write_cells(src_path, label, img_list, df_cells)


if __name__ == "__main__":
    main()
