#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>
#include "cnn.h"
#include "utils/data_loader.h"
#include "layers/conv.h"   
#include "layers/full.h"   

#define BATCH_SIZE 32
#define LEARNING_RATE 0.01f
#define EPOCHS 5

void sync_gradients(Network* net, int world_size) {
    {
        int count = net->conv1->grad_weights->total_size;
        float* local_grads = net->conv1->grad_weights->data;
        float* global_grads = (float*)malloc(count * sizeof(float));

        // MPI Allreduce: Sums all local_grads into global_grads and distributes back to everyone
        MPI_Allreduce(local_grads, global_grads, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        // Average and write back
        #pragma omp parallel for
        for (int i = 0; i < count; i++) {
            local_grads[i] = global_grads[i] / world_size;
        }
        free(global_grads);
    }

    // 2. Conv1 Biases
    {
        int count = net->conv1->grad_biases->total_size;
        float* local_grads = net->conv1->grad_biases->data;
        float* global_grads = (float*)malloc(count * sizeof(float));

        MPI_Allreduce(local_grads, global_grads, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        #pragma omp parallel for
        for (int i = 0; i < count; i++) {
            local_grads[i] = global_grads[i] / world_size;
        }
        free(global_grads);
    }

    // 3. FC1 Weights
    {
        int count = net->fc1->grad_weights->total_size;
        float* local_grads = net->fc1->grad_weights->data;
        float* global_grads = (float*)malloc(count * sizeof(float));

        MPI_Allreduce(local_grads, global_grads, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        #pragma omp parallel for
        for (int i = 0; i < count; i++) {
            local_grads[i] = global_grads[i] / world_size;
        }
        free(global_grads);
    }

    // 4. FC1 Biases
    {
        int count = net->fc1->grad_biases->total_size;
        float* local_grads = net->fc1->grad_biases->data;
        float* global_grads = (float*)malloc(count * sizeof(float));

        MPI_Allreduce(local_grads, global_grads, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        #pragma omp parallel for
        for (int i = 0; i < count; i++) {
            local_grads[i] = global_grads[i] / world_size;
        }
        free(global_grads);
    }
}

int main(int argc, char** argv) {
    // --- 1. Initialize MPI ---
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Set Random Seed (Critical: Must be different per rank if shuffling, 
    // but here we want consistent weight init if we didn't broadcast them.
    // However, our logic initializes weights independently. For true sync, 
    // we should broadcast weights from Rank 0, but for simplicity, we seed same.)
    srand(42); 

    if (rank == 0) {
        printf("==================================================\n");
        printf(" Distributed CNN Training (C + MPI + OpenMP)\n");
        printf(" Nodes: %d | OpenMP Threads: %d\n", world_size, omp_get_max_threads());
        printf("==================================================\n");
    }

    // --- 2. Load Data ---
    // Make sure you have the files in data/ directory
    // or change paths to matches your setup
    const char* img_path = "../data/custom-images-idx3-ubyte";
    const char* lbl_path = "../data/custom-labels-idx1-ubyte";

    if (rank == 0) printf("Loading Data...\n");
    
    // Each rank loads ONLY its partition of the data
    MNISTData data = load_mnist_distributed(img_path, lbl_path, rank, world_size);

    // --- 3. Build Network ---
    Network* net = build_network(BATCH_SIZE);

    // Sync Initial Weights (Optional but recommended so everyone starts same)
    // In a production system, Rank 0 would broadcast weights here.
    // For this example, relying on identical srand(42) is "good enough".

    // --- 4. Training Loop ---
    Tensor* batch_input = create_tensor(BATCH_SIZE, 1, 28, 28);
    unsigned char* batch_labels = (unsigned char*)malloc(BATCH_SIZE * sizeof(unsigned char));

    int num_batches = data.size / BATCH_SIZE;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double start_time = MPI_Wtime();
        float epoch_loss = 0.0f;
        
        for (int b = 0; b < num_batches; ++b) {
            
            // A. Prepare Batch
            // Copy data from our large loaded array into the mini-batch tensor
            int offset = b * BATCH_SIZE;
            
            // Copy Images (Parallelized copy)
            int img_size = 28 * 28;
            #pragma omp parallel for
            for (int i = 0; i < BATCH_SIZE * img_size; i++) {
                batch_input->data[i] = data.images[offset * img_size + i];
            }

            // Copy Labels
            for (int i = 0; i < BATCH_SIZE; i++) {
                batch_labels[i] = data.labels[offset + i];
            }

            // B. Local Compute (Forward + Backward)
            // Returns loss and populates gradient buffers
            float loss = forward_backward_pass(net, batch_input, batch_labels, LEARNING_RATE);
            epoch_loss += loss;

            // C. Distributed Synchronization
            // Sum gradients from all ranks and average them
            sync_gradients(net, world_size);

            // D. Optimizer Step
            // Apply the globally averaged gradients
            optimizer_step(net, LEARNING_RATE);

            // Logging (Rank 0 only)
            if (rank == 0 && b % 10 == 0) {
                printf("\rEpoch %d | Batch %d/%d | Loss: %.4f", epoch+1, b, num_batches, loss);
                fflush(stdout);
            }
        }

        double end_time = MPI_Wtime();
        
        // Aggregate total loss across all ranks for reporting
        float global_loss = 0.0f;
        MPI_Reduce(&epoch_loss, &global_loss, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            printf("\rEpoch %d Completed in %.2f sec | Avg Loss: %.4f\n", 
                   epoch+1, end_time - start_time, global_loss / (world_size * num_batches));
        }
    }

    // --- 5. Cleanup ---
    free_network(net);
    free_tensor(batch_input);
    free(batch_labels);
    free_mnist_data(&data);

    MPI_Finalize();
    return 0;
}
