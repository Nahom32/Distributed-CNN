
#ifndef FULL_H
#define FULL_H

#include "../cnn.h"

struct FCLayer {
    Tensor* weights;      // [Output_Size, Input_Size]
    Tensor* biases;       // [Output_Size]
    Tensor* grad_weights;
    Tensor* grad_biases;
    
    int input_size;
    int output_size;
};

// Function Prototypes
FCLayer* create_fc_layer(int input_size, int output_size);
void fc_forward(FCLayer* layer, Tensor* input, Tensor* output);
void fc_backward(FCLayer* layer, Tensor* input, Tensor* output_grad, Tensor* input_grad);

#endif
