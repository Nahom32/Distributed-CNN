#ifndef CONV_H
#define CONV_H

#include "../cnn.h"

void conv2d_forward(ConvLayer* layer, Tensor* input, Tensor* output);

void conv2d_backward(ConvLayer* layer, Tensor* input, Tensor* output_grad, Tensor* input_grad);
ConvLayer* create_conv_layer(int input_channels, int output_channels, int kernel_size, int input_h, int input_w);

#endif
