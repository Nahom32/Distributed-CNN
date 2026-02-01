
#ifndef FULL_H
#define FULL_H

#include "../cnn.h"

FCLayer* create_fc_layer(int input_size, int output_size);

void fc_forward(FCLayer* layer, Tensor* input, Tensor* output);

void fc_backward(FCLayer* layer, Tensor* input, Tensor* output_grad, Tensor* input_grad);

#endif
