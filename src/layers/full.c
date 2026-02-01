#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "layers/full.h"

// Helper: Xavier Initialization for FC weights
static void init_fc_weights(Tensor* t, int fan_in, int fan_out) {
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    for (int i = 0; i < t->total_size; ++i) {
        float r = (float)rand() / (float)RAND_MAX;
        t->data[i] = (r * 2.0f * limit) - limit;
    }
}

FCLayer* create_fc_layer(int input_size, int output_size) {
    FCLayer* layer = (FCLayer*)malloc(sizeof(FCLayer));
    layer->input_size = input_size;
    layer->output_size = output_size;

    // Weights: [Output_Size, Input_Size] (Transposed logic often easier for dot products)
    // We create a tensor [1, 1, Output, Input] just to use the struct
    layer->weights = create_tensor(1, 1, output_size, input_size);
    layer->grad_weights = create_tensor(1, 1, output_size, input_size);
    
    // Biases: [Output_Size]
    layer->biases = create_tensor(1, 1, 1, output_size);
    layer->grad_biases = create_tensor(1, 1, 1, output_size);

    init_fc_weights(layer->weights, input_size, output_size);

    // Initialize biases to 0
    for(int i=0; i<layer->biases->total_size; i++) layer->biases->data[i] = 0.0f;

    return layer;
}

// ------------------------------------------------------------------
// FORWARD PASS
// Output[b, out] = Sum(Input[b, in] * Weights[out, in]) + Bias[out]
// ------------------------------------------------------------------
void fc_forward(FCLayer* layer, Tensor* input, Tensor* output) {
    int batch = input->b;
    int in_size = layer->input_size;
    int out_size = layer->output_size;

    // Sanity check: Input total dimensions must match expected input size
    if ((input->c * input->h * input->w) != in_size) {
        fprintf(stderr, "Error: FC Layer expected input size %d, got %d\n", 
                in_size, input->c * input->h * input->w);
        exit(1);
    }

    // Reset output
    // #pragma omp parallel for
    for(int i=0; i<output->total_size; i++) output->data[i] = 0.0f;

    // Parallelize over the Batch
    #pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        
        // For each output neuron
        for (int o = 0; o < out_size; ++o) {
            
            float sum = 0.0f;
            float bias = layer->biases->data[o];

            // Dot Product
            // Optimization: This inner loop can be vectorized by the compiler (SIMD)
            for (int i = 0; i < in_size; ++i) {
                // Input Index: Row 'b', Col 'i'
                int in_idx = b * in_size + i;
                
                // Weight Index: Row 'o', Col 'i'
                int w_idx = o * in_size + i;

                sum += input->data[in_idx] * layer->weights->data[w_idx];
            }
            
            // Output Index: Row 'b', Col 'o'
            output->data[b * out_size + o] = sum + bias;
        }
    }
}

// ------------------------------------------------------------------
// BACKWARD PASS
// 1. dW = Output_Grad^T * Input
// 2. dB = Sum(Output_Grad, axis=0)
// 3. dInput = Output_Grad * W^T
// ------------------------------------------------------------------
void fc_backward(FCLayer* layer, Tensor* input, Tensor* output_grad, Tensor* input_grad) {
    int batch = input->b;
    int in_size = layer->input_size;
    int out_size = layer->output_size;

    // 1. Clear Gradients
    for(int i=0; i<layer->grad_weights->total_size; i++) layer->grad_weights->data[i] = 0.0f;
    for(int i=0; i<layer->grad_biases->total_size; i++) layer->grad_biases->data[i] = 0.0f;
    if (input_grad) {
        for(int i=0; i<input_grad->total_size; i++) input_grad->data[i] = 0.0f;
    }

    // 2. Compute Gradient of Weights and Biases
    // Parallelize over Output neurons
    #pragma omp parallel for
    for (int o = 0; o < out_size; ++o) {
        
        // For biases: Sum gradients across the batch
        float db_sum = 0.0f;

        for (int b = 0; b < batch; ++b) {
            float grad_out = output_grad->data[b * out_size + o];
            db_sum += grad_out;

            // For weights: Accumulate Input * Grad_Out
            for (int i = 0; i < in_size; ++i) {
                float in_val = input->data[b * in_size + i];
                
                // Weight Gradient Index: [o, i]
                // We use atomic add if we parallelize over batch, but here we parallelize over 'o'
                // so each thread owns a specific row of weights. No atomic needed!
                layer->grad_weights->data[o * in_size + i] += in_val * grad_out;
            }
        }
        layer->grad_biases->data[o] = db_sum;
    }

    // 3. Compute Gradient of Input (Pass back to previous layer)
    if (input_grad) {
        // Parallelize over Batch
        #pragma omp parallel for
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < in_size; ++i) {
                
                float sum = 0.0f;
                // Dot product of (Grad_Out vector) * (Weights column i)
                for (int o = 0; o < out_size; ++o) {
                    float grad_out = output_grad->data[b * out_size + o];
                    float w_val = layer->weights->data[o * in_size + i];
                    sum += grad_out * w_val;
                }
                
                input_grad->data[b * in_size + i] = sum;
            }
        }
    }
}
