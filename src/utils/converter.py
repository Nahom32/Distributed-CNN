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
    # 1. Write Images (Magic Number: 2051)
    # Header: Magic(4), Count(4), Rows(4), Cols(4)
    header = struct.pack(">IIII", 2051, len(images), 28, 28)

    # Flatten data and ensure it is uint8 (0-255)
    img_data = np.array([np.array(img) for img in images], dtype=np.uint8).flatten()

    img_filename = f"{prefix}-images-idx3-ubyte"
    with open(img_filename, "wb") as f:
        f.write(header)
        f.write(img_data.tobytes())
    print(f"Saved Binary Images: {img_filename}")

    # 2. Write Labels (Magic Number: 2049)
    # Header: Magic(4), Count(4)
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

    # Create output directory for PNGs
    if SAVE_PNGS and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    images_buffer = []
    labels_buffer = []

    print("Processing images...")

    # Inspect columns to find image/label data
    # HuggingFace parquet usually has 'image' (dict or bytes) and 'label' (int)
    for index, row in df.iterrows():
        # --- Handle Image ---
        img_entry = row["image"]
        label = row["label"]

        # If image is a dictionary (HuggingFace format), extract bytes
        if isinstance(img_entry, dict) and "bytes" in img_entry:
            image = Image.open(io.BytesIO(img_entry["bytes"]))
        elif isinstance(img_entry, bytes):
            image = Image.open(io.BytesIO(img_entry))
        else:
            # Assume it's already a PIL object or raw array
            image = img_entry

        # Ensure 28x28 Grayscale
        image = image.resize((28, 28)).convert("L")

        # Buffer for Binary Export
        if SAVE_IDX_BINARY:
            images_buffer.append(image)
            labels_buffer.append(label)

        # Save as PNG (Regular Image)
        if SAVE_PNGS:
            if MAX_PNGS is None or index < MAX_PNGS:
                image.save(f"{OUTPUT_DIR}/img_{index}_lbl_{label}.png")

    print(f"Processed {len(df)} records.")

    if SAVE_IDX_BINARY:
        print("Converting to IDX Binary format for C code...")
        write_idx_files(images_buffer, labels_buffer)


if __name__ == "__main__":
    import io  # Needed for byte stream reading

    process_parquet()
