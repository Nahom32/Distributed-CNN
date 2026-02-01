#include <float.h>
#include <stdio.h>
#include <omp.h>
#include "layers/pool.h"

// ------------------------------------------------------------------
// FORWARD PASS
// Logic: Slide a window (kernel_size) over the input. 
// Output pixel = Maximum value in that window.
// ------------------------------------------------------------------
void max_pool_forward(Tensor* input, Tensor* output, int kernel_size, int stride) {
    int batch = input->b;
    int channels = input->c;
    int out_h = output->h;
    int out_w = output->w;

    // Parallelize over Batch and Channels (Completely independent)
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            
            // Iterate over the Output Grid
            for (int i = 0; i < out_h; ++i) {
                for (int j = 0; j < out_w; ++j) {
                    
                    // Calculate the top-left corner of the input window
                    int h_start = i * stride;
                    int w_start = j * stride;
                    int h_end = h_start + kernel_size;
                    int w_end = w_start + kernel_size;

                    // Find Max in this window
                    float max_val = -FLT_MAX;

                    for (int y = h_start; y < h_end; ++y) {
                        for (int x = w_start; x < w_end; ++x) {
                            
                            // Bounds checking (in case of odd sizes)
                            if (y < input->h && x < input->w) {
                                int in_idx = b * (channels * input->h * input->w) +
                                             c * (input->h * input->w) +
                                             y * input->w + x;
                                
                                float val = input->data[in_idx];
                                if (val > max_val) {
                                    max_val = val;
                                }
                            }
                        }
                    }

                    // Write to Output
                    int out_idx = b * (channels * out_h * out_w) +
                                  c * (out_h * out_w) +
                                  i * out_w + j;
                                  
                    output->data[out_idx] = max_val;
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// BACKWARD PASS
// Logic: The gradient only flows back to the pixel that "won" (was max)
// during the forward pass. All other pixels get 0 gradient.
// ------------------------------------------------------------------
void max_pool_backward(Tensor* input, Tensor* output_grad, Tensor* input_grad, int kernel_size, int stride) {
    int batch = input->b;
    int channels = input->c;
    int out_h = output_grad->h;
    int out_w = output_grad->w;

    // 1. Reset Input Gradients to 0
    // #pragma omp parallel for
    for(int i=0; i<input_grad->total_size; i++) input_grad->data[i] = 0.0f;

    // 2. Distribute Gradients
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            
            for (int i = 0; i < out_h; ++i) {
                for (int j = 0; j < out_w; ++j) {
                    
                    int h_start = i * stride;
                    int w_start = j * stride;
                    int h_end = h_start + kernel_size;
                    int w_end = w_start + kernel_size;

                    // Get the gradient value coming from above
                    int out_idx = b * (channels * out_h * out_w) +
                                  c * (out_h * out_w) +
                                  i * out_w + j;
                    float grad_val = output_grad->data[out_idx];

                    // Find the Max Index again (Recalculation Strategy)
                    int max_idx = -1;
                    float max_val = -FLT_MAX;

                    for (int y = h_start; y < h_end; ++y) {
                        for (int x = w_start; x < w_end; ++x) {
                             if (y < input->h && x < input->w) {
                                int in_idx = b * (channels * input->h * input->w) +
                                             c * (input->h * input->w) +
                                             y * input->w + x;
                                
                                if (input->data[in_idx] > max_val) {
                                    max_val = input->data[in_idx];
                                    max_idx = in_idx;
                                }
                             }
                        }
                    }

                    // Pass the gradient ONLY to the max pixel
                    if (max_idx != -1) {
                        // We use atomic add just in case of overlapping windows (stride < kernel_size),
                        // though for standard MaxPool (stride=2, kernel=2) they don't overlap.
                        // Atomic is safer for general cases.
                        #pragma omp atomic
                        input_grad->data[max_idx] += grad_val;
                    }
                }
            }
        }
    }
}
