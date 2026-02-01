#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "layers/conv.h"

// Helper: Initialize weights with Xavier Initialization
void init_weights(Tensor* t, int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    for (int i = 0; i < t->total_size; ++i) {
        // Generate random float between -limit and limit
        float r = (float)rand() / (float)RAND_MAX;
        t->data[i] = (r * 2.0f * limit) - limit;
    }
}

ConvLayer* create_conv_layer(int in_c, int out_c, int k_size, int in_h, int in_w) {
    ConvLayer* layer = (ConvLayer*)malloc(sizeof(ConvLayer));
    
    layer->input_channels = in_c;
    layer->output_channels = out_c;
    layer->kernel_size = k_size;
    
    int out_h = in_h - k_size + 1; // Assuming stride=1, padding=0
    int out_w = in_w - k_size + 1;

    layer->weights = create_tensor(out_c, in_c, k_size, k_size);
    layer->grad_weights = create_tensor(out_c, in_c, k_size, k_size);
    
    // Biases: [Out_Channels]
    layer->biases = create_tensor(1, out_c, 1, 1);
    layer->grad_biases = create_tensor(1, out_c, 1, 1);

    // Initialize
    int fan_in = in_c * k_size * k_size;
    int fan_out = out_c * k_size * k_size;
    init_weights(layer->weights, fan_in, fan_out);
    
    // Zero out biases
    for(int i=0; i<layer->biases->total_size; i++) layer->biases->data[i] = 0.0f;

    return layer;
}

// ------------------------------------------------------------------
// FORWARD PASS
// Formula: Output[b, k, i, j] = Sum(Input[b, c, i+m, j+n] * Weight[k, c, m, n]) + Bias[k]
// ------------------------------------------------------------------
void conv2d_forward(ConvLayer* layer, Tensor* input, Tensor* output) {
    int batch = input->b;
    int out_c = layer->output_channels;
    int in_c = layer->input_channels;
    int k_size = layer->kernel_size;
    int out_h = output->h;
    int out_w = output->w;

    // #pragma omp parallel for
    for(int i=0; i<output->total_size; i++) output->data[i] = 0.0f;

    // Parallelize over Batch and Output Channels (The "Independent" units)
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int k = 0; k < out_c; ++k) {
            
            // Get Bias for this filter
            float bias_val = layer->biases->data[k];

            // Iterate over output spatial dimensions
            for (int i = 0; i < out_h; ++i) {
                for (int j = 0; j < out_w; ++j) {
                    
                    float sum = 0.0f;

                    // Convolution: Dot product of Kernel and Input Patch
                    for (int c = 0; c < in_c; ++c) {
                        for (int m = 0; m < k_size; ++m) {
                            for (int n = 0; n < k_size; ++n) {
                                
                                // Input Index: [b, c, i+m, j+n]
                                int input_idx = b * (in_c * input->h * input->w) +
                                                c * (input->h * input->w) +
                                                (i + m) * input->w +
                                                (j + n);
                                
                                // Weight Index: [k, c, m, n]
                                int weight_idx = k * (in_c * k_size * k_size) +
                                                 c * (k_size * k_size) +
                                                 m * k_size + 
                                                 n;

                                sum += input->data[input_idx] * layer->weights->data[weight_idx];
                            }
                        }
                    }
                    
                    // Output Index: [b, k, i, j]
                    int out_idx = b * (out_c * out_h * out_w) +
                                  k * (out_h * out_w) +
                                  i * out_w + 
                                  j;

                    output->data[out_idx] = sum + bias_val;
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// BACKWARD PASS
// 1. dL/dW = Input * dL/dOutput (Convolution of Input and Output Gradients)
// 2. dL/dX = dL/dOutput * W (Full convolution, usually via padding/rotation)
// ------------------------------------------------------------------
void conv2d_backward(ConvLayer* layer, Tensor* input, Tensor* output_grad, Tensor* input_grad) {
    int batch = input->b;
    int out_c = layer->output_channels;
    int in_c = layer->input_channels;
    int k_size = layer->kernel_size;
    int out_h = output_grad->h;
    int out_w = output_grad->w;

    // 1. Clear Gradients
    for(int i=0; i<layer->grad_weights->total_size; i++) layer->grad_weights->data[i] = 0.0f;
    for(int i=0; i<layer->grad_biases->total_size; i++) layer->grad_biases->data[i] = 0.0f;
    if(input_grad) {
        for(int i=0; i<input_grad->total_size; i++) input_grad->data[i] = 0.0f;
    }

    // 2. Compute Gradients for Weights and Biases
    // Parallelize over Output Channels and Input Channels
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < out_c; ++k) {
        for (int c = 0; c < in_c; ++c) {
            
            // For this filter connection k->c, accumulate over the whole batch
            for (int b = 0; b < batch; ++b) {
                
                // Convolve Input with Output Gradient
                for (int m = 0; m < k_size; ++m) {
                    for (int n = 0; n < k_size; ++n) {
                        
                        float dw_sum = 0.0f;

                        for (int i = 0; i < out_h; ++i) {
                            for (int j = 0; j < out_w; ++j) {
                                
                                int in_idx = b * (in_c * input->h * input->w) +
                                             c * (input->h * input->w) +
                                             (i + m) * input->w + (j + n);
                                
                                int grad_idx = b * (out_c * out_h * out_w) +
                                               k * (out_h * out_w) +
                                               i * out_w + j;

                                dw_sum += input->data[in_idx] * output_grad->data[grad_idx];
                            }
                        }
                        
                        // Accumulate into Weight Gradient Tensor
                        // Use atomic because other threads might process different batches 
                        // (though here we put batch in inner loop, so atomic strictly not needed for this structure, 
                        // but good practice if you re-order loops)
                        int w_idx = k * (in_c * k_size * k_size) + c * (k_size * k_size) + m * k_size + n;
                        
                        #pragma omp atomic
                        layer->grad_weights->data[w_idx] += dw_sum;
                    }
                }

                // Bias Gradient: Just sum the Output Gradients
                float db_sum = 0.0f;
                for (int i = 0; i < out_h; ++i) {
                    for (int j = 0; j < out_w; ++j) {
                        int grad_idx = b * (out_c * out_h * out_w) + k * (out_h * out_w) + i * out_w + j;
                        db_sum += output_grad->data[grad_idx];
                    }
                }
                #pragma omp atomic
                layer->grad_biases->data[k] += db_sum;
            }
        }
    }

    // 3. Compute Gradients for Input (Pass down to previous layer)
    // Only needed if this isn't the very first layer
    if (input_grad) {
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < in_c; ++c) {
                
                for (int k = 0; k < out_c; ++k) {
                    for (int i = 0; i < out_h; ++i) {
                        for (int j = 0; j < out_w; ++j) {
                            
                            int grad_idx = b * (out_c * out_h * out_w) + k * (out_h * out_w) + i * out_w + j;
                            float grad_val = output_grad->data[grad_idx];

                            for (int m = 0; m < k_size; ++m) {
                                for (int n = 0; n < k_size; ++n) {
                                    
                                    int w_idx = k * (in_c * k_size * k_size) + c * (k_size * k_size) + m * k_size + n;
                                    int in_idx = b * (in_c * input->h * input->w) + c * (input->h * input->w) + (i + m) * input->w + (j + n);

                                    // Atomic add because patches overlap
                                    #pragma omp atomic
                                    input_grad->data[in_idx] += grad_val * layer->weights->data[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
