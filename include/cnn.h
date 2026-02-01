#ifndef CNN_H
#define CNN_H

#include <stdlib.h>

typedef struct {
    float* data;
    int b, c, h, w;
    size_t total_size;
} Tensor;

Tensor* create_tensor(int b, int c, int h, int w);
void free_tensor(Tensor* t);

typedef struct ConvLayer ConvLayer; 
typedef struct FCLayer FCLayer;     

typedef struct {
    struct ConvLayer* conv1;
    struct FCLayer* fc1;

    Tensor* input;          // The input image batch
    Tensor* t_conv1_out;    // Output of Conv1
    Tensor* t_relu_out;     // Output of ReLU
    Tensor* t_pool_out;     // Output of MaxPool
    Tensor* t_fc_out;       // Final Logits (Pre-Softmax)
    
    Tensor* t_fc_grad;      // dLoss/dOutput
    Tensor* t_pool_grad;    // Gradient flowing into Pool
    Tensor* t_relu_grad;    // Gradient flowing into ReLU
    Tensor* t_conv1_grad;   // Gradient flowing into Conv1
} Network;

Network* build_network(int batch_size);
void free_network(Network* net);

float forward_backward_pass(Network* net, Tensor* input_data, unsigned char* labels, float learning_rate);

void optimizer_step(Network* net, float learning_rate);
int evaluate_accuracy(Network* net, Tensor* input_data, unsigned char* labels, int count);
#endif
