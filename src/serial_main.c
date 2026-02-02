#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cnn.h"
#include "layers/conv.h"
#include "layers/full.h"
#include "utils/data_loader.h"

#define BATCH_SIZE 32
#define LEARNING_RATE 0.01f

int main(int argc, char** argv) {
    int epochs = 5;
    if (argc > 1) {
        epochs = atoi(argv[1]);
    }

    printf("==================================================\n");
    printf(" Serial CNN Training (No MPI, No OpenMP)\n");
    printf("==================================================\n");

    const char* tr_img = "../data/custom-images-idx3-ubyte";
    const char* tr_lbl = "../data/custom-labels-idx1-ubyte";
    MNISTData train_data = load_mnist_distributed(tr_img, tr_lbl, 0, 1);

    const char* te_img = "../data/custom_test-images-idx3-ubyte";
    const char* te_lbl = "../data/custom_test-labels-idx1-ubyte";
    MNISTData test_data = load_mnist_distributed(te_img, te_lbl, 0, 1);

    Network* net = build_network(BATCH_SIZE);
    
    Tensor* batch_input = create_tensor(BATCH_SIZE, 1, 28, 28);
    unsigned char* batch_labels = (unsigned char*)malloc(BATCH_SIZE);

    Tensor* test_input_tensor = create_tensor(test_data.size, 1, 28, 28);
    for(int i=0; i<test_data.size * 784; i++) {
        test_input_tensor->data[i] = test_data.images[i];
    }

    clock_t start_time = clock(); // Standard C timing

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int num_batches = train_data.size / BATCH_SIZE;
        
        for (int b = 0; b < num_batches; ++b) {
            int offset = b * BATCH_SIZE;
            
            for (int i = 0; i < BATCH_SIZE * 784; i++) {
                batch_input->data[i] = train_data.images[offset * 784 + i];
            }
            for (int i = 0; i < BATCH_SIZE; i++) {
                batch_labels[i] = train_data.labels[offset + i];
            }

            float loss = forward_backward_pass(net, batch_input, batch_labels, LEARNING_RATE);
            epoch_loss += loss;

            optimizer_step(net, LEARNING_RATE);
        }

        int correct = evaluate_accuracy(net, test_input_tensor, test_data.labels, test_data.size);
        float acc = (float)correct / test_data.size;
        float avg_loss = epoch_loss / num_batches;

        printf("Epoch %d | Loss: %.4f | Test Acc: %.4f%%\n", epoch+1, avg_loss, acc * 100.0f);
    }

    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Total Serial Execution Time: %.2f seconds\n", time_taken);

    free_network(net);
    free_tensor(batch_input);
    free_tensor(test_input_tensor);
    free(batch_labels);
    free_mnist_data(&train_data);
    free_mnist_data(&test_data);
    
    return 0;
}
