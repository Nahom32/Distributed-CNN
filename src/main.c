#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <time.h>
#include "cnn.h"
#include "layers/conv.h"
#include "layers/full.h"
#include "utils/data_loader.h"

#define BATCH_SIZE 32
#define LEARNING_RATE 0.01f

void sync_gradients(Network* net, int world_size) {
    // 1. Conv1 Weights
    MPI_Allreduce(MPI_IN_PLACE, net->conv1->grad_weights->data, net->conv1->grad_weights->total_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, net->conv1->grad_biases->data, net->conv1->grad_biases->total_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    // 2. FC1 Weights
    MPI_Allreduce(MPI_IN_PLACE, net->fc1->grad_weights->data, net->fc1->grad_weights->total_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, net->fc1->grad_biases->data, net->fc1->grad_biases->total_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    
    // Average
    float scale = 1.0f / world_size;
    int s1 = net->conv1->grad_weights->total_size;
    #pragma omp parallel for
    for(int i=0; i<s1; i++) net->conv1->grad_weights->data[i] *= scale;
    
    int s2 = net->conv1->grad_biases->total_size;
    #pragma omp parallel for
    for(int i=0; i<s2; i++) net->conv1->grad_biases->data[i] *= scale;

    int s3 = net->fc1->grad_weights->total_size;
    #pragma omp parallel for
    for(int i=0; i<s3; i++) net->fc1->grad_weights->data[i] *= scale;

    int s4 = net->fc1->grad_biases->total_size;
    #pragma omp parallel for
    for(int i=0; i<s4; i++) net->fc1->grad_biases->data[i] *= scale;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 1. Parse Epochs from Command Line
    int epochs = 5; // Default
    if (argc > 1) {
        epochs = atoi(argv[1]);
    }

    // 2. Load Training Data
    const char* tr_img = "../data/custom-images-idx3-ubyte";
    const char* tr_lbl = "../data/custom-labels-idx1-ubyte";
    MNISTData train_data = load_mnist_distributed(tr_img, tr_lbl, rank, world_size);

    const char* te_img = "../data/custom_test-images-idx3-ubyte";
    const char* te_lbl = "../data/custom_test-labels-idx1-ubyte";
    MNISTData test_data = load_mnist_distributed(te_img, te_lbl, rank, world_size);

    Network* net = build_network(BATCH_SIZE);
    
    // Setup Training Batch Tensors
    Tensor* batch_input = create_tensor(BATCH_SIZE, 1, 28, 28);
    unsigned char* batch_labels = (unsigned char*)malloc(BATCH_SIZE);
    
    // Prepare temporary Tensor wrapper for the full local dataset (for evaluation)
    Tensor* local_test_tensor = create_tensor(test_data.size, 1, 28, 28);
    // Copy data once to avoid recopying during eval loop
    #pragma omp parallel for
    for(int i=0; i<test_data.size * 784; i++) local_test_tensor->data[i] = test_data.images[i];


    double total_start_time = MPI_Wtime();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int num_batches = train_data.size / BATCH_SIZE;
        
        // --- TRAINING LOOP ---
        for (int b = 0; b < num_batches; ++b) {
            int offset = b * BATCH_SIZE;
            
            // Load Batch
            #pragma omp parallel for
            for (int i = 0; i < BATCH_SIZE * 784; i++) {
                batch_input->data[i] = train_data.images[offset * 784 + i];
            }
            for (int i = 0; i < BATCH_SIZE; i++) batch_labels[i] = train_data.labels[offset + i];

            // Train Step
            epoch_loss += forward_backward_pass(net, batch_input, batch_labels, LEARNING_RATE);
            sync_gradients(net, world_size);
            optimizer_step(net, LEARNING_RATE);
        }

        // --- EVALUATION (TEST ACCURACY) ---
        // 1. Calculate Local Accuracy
        int local_correct = evaluate_accuracy(net, local_test_tensor, test_data.labels, test_data.size);
        int local_total = test_data.size;

        // 2. Reduce to Global Accuracy
        int global_correct = 0;
        int global_total_test_samples = 0;

        MPI_Reduce(&local_correct, &global_correct, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_total, &global_total_test_samples, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        // 3. Aggregate Loss
        float global_loss = 0.0f;
        MPI_Reduce(&epoch_loss, &global_loss, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        // 4. Log Metrics (Only Rank 0 prints)
        if (rank == 0) {
            float avg_loss = global_loss / (world_size * num_batches);
            float accuracy = (float)global_correct / (float)global_total_test_samples;
            
            // Print readable log
            printf("Epoch %d | Loss: %.4f | Test Acc: %.4f%%\n", epoch+1, avg_loss, accuracy * 100.0f);
            
            // Print Parsable Log for Python (PREFIX: METRICS)
            printf("METRICS,%d,%.4f,%.4f\n", epoch+1, avg_loss, accuracy);
            fflush(stdout);
        }
    }

    double total_end_time = MPI_Wtime();
    if (rank == 0) {
        printf("TIME_TOTAL,%f\n", total_end_time - total_start_time);
    }

    // Cleanup
    free_network(net);
    free_tensor(batch_input);
    free_tensor(local_test_tensor);
    free(batch_labels);
    free_mnist_data(&train_data);
    free_mnist_data(&test_data);
    
    MPI_Finalize();
    return 0;
}
