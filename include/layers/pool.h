#ifndef POOL_H
#define POOL_H

#include "../cnn.h"

void max_pool_forward(Tensor* input, Tensor* output, int kernel_size, int stride);

void max_pool_backward(Tensor* input, Tensor* output_grad, Tensor* input_grad, int kernel_size, int stride);

#endif
