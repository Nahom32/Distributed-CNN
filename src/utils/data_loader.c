#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data_loader.h"

#define IMAGES_HEADER_SIZE 16
#define LABELS_HEADER_SIZE 8

static uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val << 8)  & 0x00ff0000) |
           ((val >> 8)  & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

MNISTData load_mnist_distributed(const char* image_filename, 
                                 const char* label_filename, 
                                 int rank, 
                                 int world_size) {
    MNISTData data = {0};
    FILE *f_img, *f_lbl;

    // --- 1. Open Files ---
    if (!(f_img = fopen(image_filename, "rb"))) {
        fprintf(stderr, "Rank %d: Could not open image file: %s\n", rank, image_filename);
        exit(1);
    }
    if (!(f_lbl = fopen(label_filename, "rb"))) {
        fprintf(stderr, "Rank %d: Could not open label file: %s\n", rank, label_filename);
        exit(1);
    }

    // --- 2. Parse Image Header ---
    uint32_t magic, num_images, rows, cols;
    fread(&magic, 4, 1, f_img);
    fread(&num_images, 4, 1, f_img);
    fread(&rows, 4, 1, f_img);
    fread(&cols, 4, 1, f_img);

    num_images = swap_endian(num_images);
    rows = swap_endian(rows);
    cols = swap_endian(cols);
    data.rows = (int)rows;
    data.cols = (int)cols;

    // --- 3. Parse Label Header ---
    uint32_t magic_lbl, num_labels;
    fread(&magic_lbl, 4, 1, f_lbl);
    fread(&num_labels, 4, 1, f_lbl);
    num_labels = swap_endian(num_labels);

    // Safety Check
    if (num_images != num_labels) {
        fprintf(stderr, "Error: Image count (%d) and Label count (%d) mismatch.\n", num_images, num_labels);
        exit(1);
    }

    // --- 4. Calculate Partition (MPI Split) ---
    // Total dataset size is 60,000.
    // Rank 0 gets 0-14999, Rank 1 gets 15000-29999, etc.
    int local_count = num_images / world_size;
    int remainder = num_images % world_size;

    // Handle remainder (distribute extra items to first few ranks)
    int start_index = rank * local_count + (rank < remainder ? rank : remainder);
    if (rank < remainder) {
        local_count++;
    }

    data.size = local_count;
    int image_size = rows * cols; 

    printf("[Rank %d] Loading %d images (Index: %d to %d)\n", 
           rank, local_count, start_index, start_index + local_count - 1);

    // --- 5. Seek to Partition Start ---
    // Images: Header (16 bytes) + (Start_Index * 784 bytes)
    long img_offset = IMAGES_HEADER_SIZE + ((long)start_index * image_size);
    fseek(f_img, img_offset, SEEK_SET);

    // Labels: Header (8 bytes) + (Start_Index * 1 byte)
    long lbl_offset = LABELS_HEADER_SIZE + ((long)start_index);
    fseek(f_lbl, lbl_offset, SEEK_SET);

    // --- 6. Allocate Memory ---
    // Allocate temporary buffer for raw bytes (uint8)
    unsigned char* raw_img_buffer = (unsigned char*)malloc(local_count * image_size);
    data.labels = (uint8_t*)malloc(local_count * sizeof(uint8_t));
    
    // Allocate final float buffer for normalized data
    data.images = (float*)malloc(local_count * image_size * sizeof(float));

    if (!raw_img_buffer || !data.labels || !data.images) {
        fprintf(stderr, "Rank %d: Memory allocation failed.\n", rank);
        exit(1);
    }

    // --- 7. Read Data ---
    fread(raw_img_buffer, 1, local_count * image_size, f_img);
    fread(data.labels, 1, local_count, f_lbl);

    // --- 8. Normalize Data (0-255 -> 0.0-1.0) ---
    // This is computationally heavy, so we can use OpenMP here too!
    #pragma omp parallel for
    for (int i = 0; i < local_count * image_size; i++) {
        data.images[i] = (float)raw_img_buffer[i] / 255.0f;
    }

    // --- 9. Cleanup ---
    free(raw_img_buffer);
    fclose(f_img);
    fclose(f_lbl);

    return data;
}

void free_mnist_data(MNISTData* data) {
    if (data->images) free(data->images);
    if (data->labels) free(data->labels);
    data->images = NULL;
    data->labels = NULL;
    data->size = 0;
}
