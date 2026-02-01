import pandas as pd
import numpy as np
from PIL import Image
import os
import struct
from config import *


def write_idx_files(images, labels, prefix="custom"):
    """
    Converts raw numpy arrays into the IDX format expected by the C loader.
    """
    header = struct.pack(">IIII", 2051, len(images), 28, 28)

    img_data = np.array([np.array(img) for img in images], dtype=np.uint8).flatten()

    img_filename = f"{prefix}-images-idx3-ubyte"
    with open(img_filename, "wb") as f:
        f.write(header)
        f.write(img_data.tobytes())
    print(f"Saved Binary Images: {img_filename}")

    lbl_header = struct.pack(">II", 2049, len(labels))
    lbl_data = np.array(labels, dtype=np.uint8)

    lbl_filename = f"{prefix}-labels-idx1-ubyte"
    with open(lbl_filename, "wb") as f:
        f.write(lbl_header)
        f.write(lbl_data.tobytes())
    print(f"Saved Binary Labels: {lbl_filename}")


def process_parquet():
    print(f"Reading {INPUT_FILE}...")
    try:
        df = pd.read_parquet(INPUT_FILE)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    if SAVE_PNGS and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    images_buffer = []
    labels_buffer = []

    print("Processing images...")

    for index, row in df.iterrows():
        img_entry = row["image"]
        label = row["label"]

        if isinstance(img_entry, dict) and "bytes" in img_entry:
            image = Image.open(io.BytesIO(img_entry["bytes"]))
        elif isinstance(img_entry, bytes):
            image = Image.open(io.BytesIO(img_entry))
        else:
            image = img_entry

        image = image.resize((28, 28)).convert("L")

        if SAVE_IDX_BINARY:
            images_buffer.append(image)
            labels_buffer.append(label)

        if SAVE_PNGS:
            if MAX_PNGS is None or index < MAX_PNGS:
                image.save(f"{OUTPUT_DIR}/img_{index}_lbl_{label}.png")

    print(f"Processed {len(df)} records.")

    if SAVE_IDX_BINARY:
        print("Converting to IDX Binary format for C code...")
        write_idx_files(images_buffer, labels_buffer)


if __name__ == "__main__":
    import io

    process_parquet()
