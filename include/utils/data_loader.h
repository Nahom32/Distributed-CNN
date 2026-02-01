#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <stdint.h>

typedef struct {
    float* images;      // Normalized pixel data (0.0 - 1.0)
    uint8_t* labels;    // Class labels (0-9)
    int size;           // Number of items in this local partition
    int rows;           // Image height (28)
    int cols;           // Image width (28)
} MNISTData;

/**
 * Loads a partition of the MNIST dataset specific to the calling MPI rank.
 * * @param image_filename Path to train-images-idx3-ubyte
 * @param label_filename Path to train-labels-idx1-ubyte
 * @param rank           Current MPI rank (process ID)
 * @param world_size     Total number of MPI processes
 * @return MNISTData     Structure containing ONLY this rank's data
 */
MNISTData load_mnist_distributed(const char* image_filename, 
                                 const char* label_filename, 
                                 int rank, 
                                 int world_size);

void free_mnist_data(MNISTData* data);

#endif
