#ifndef CONV_H
#define CONV_H

#include "../cnn.h"

// Definition of the Convolutional Layer Struct
struct ConvLayer {
    Tensor* weights;      // [Output_Channels, Input_Channels, Kernel, Kernel]
    Tensor* biases;       // [Output_Channels]
    Tensor* grad_weights; // Gradients D_Loss/D_Weights
    Tensor* grad_biases;  // Gradients D_Loss/D_Biases
    
    int input_channels;
    int output_channels;
    int kernel_size;
};

// Function Prototypes
void conv2d_forward(ConvLayer* layer, Tensor* input, Tensor* output);
void conv2d_backward(ConvLayer* layer, Tensor* input, Tensor* output_grad, Tensor* input_grad);
ConvLayer* create_conv_layer(int input_channels, int output_channels, int kernel_size, int input_h, int input_w);

#endif
