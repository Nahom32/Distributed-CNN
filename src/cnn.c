#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#include "cnn.h"
#include "layers/conv.h"
#include "layers/pool.h"
#include "layers/full.h"

Tensor* create_tensor(int b, int c, int h, int w) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->b = b; t->c = c; t->h = h; t->w = w;
    t->total_size = (size_t)b * c * h * w;
    t->data = (float*)calloc(t->total_size, sizeof(float)); // Init to 0
    return t;
}

void free_tensor(Tensor* t) {
    if (t) { free(t->data); free(t); }
}

void relu_forward(Tensor* input, Tensor* output) {
    #pragma omp parallel for
    for (int i = 0; i < input->total_size; i++) {
        output->data[i] = (input->data[i] > 0.0f) ? input->data[i] : 0.0f;
    }
}

void relu_backward(Tensor* input, Tensor* output_grad, Tensor* input_grad) {
    #pragma omp parallel for
    for (int i = 0; i < input->total_size; i++) {
        // Gradient passes through if input > 0, else 0
        input_grad->data[i] = (input->data[i] > 0.0f) ? output_grad->data[i] : 0.0f;
    }
}

float softmax_loss_backward(Tensor* logits, unsigned char* labels, Tensor* grad_input) {
    int batch = logits->b;
    int classes = logits->w; // FC output is [Batch, 1, 1, 10]
    float total_loss = 0.0f;

    for (int b = 0; b < batch; b++) {
        
        float max_val = -1e9;
        for (int c = 0; c < classes; c++) {
            float val = logits->data[b * classes + c];
            if (val > max_val) max_val = val;
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < classes; c++) {
            float val = expf(logits->data[b * classes + c] - max_val);
            grad_input->data[b * classes + c] = val;
            sum_exp += val;
        }

        int correct_label = (int)labels[b];
        
        for (int c = 0; c < classes; c++) {
            float prob = grad_input->data[b * classes + c] / sum_exp;
            
            if (c == correct_label) {
                total_loss += -logf(prob + 1e-7f); // Add epsilon to prevent log(0)
                grad_input->data[b * classes + c] = (prob - 1.0f) / batch; // Normalize by batch size
            } else {
                grad_input->data[b * classes + c] = prob / batch;
            }
        }
    }
    return total_loss / batch;
}

// ------------------------------------------------------------------
Network* build_network(int batch_size) {
    Network* net = (Network*)malloc(sizeof(Network));
    
    int in_h = 28, in_w = 28, in_c = 1;
    
    net->conv1 = create_conv_layer(in_c, 8, 3, in_h, in_w);
    
    int pool_out_h = 13;
    int pool_out_w = 13;
    
    // Layer 3: Fully Connected
    // Input Flattened: 8 * 13 * 13 = 1352
    int fc_in_size = 8 * pool_out_h * pool_out_w;
    int fc_out_size = 10;
    net->fc1 = create_fc_layer(fc_in_size, fc_out_size);

    // --- 2. Allocate Intermediate Buffers ---
    // We allocate these ONCE to avoid fragmentation
    net->input       = create_tensor(batch_size, 1, 28, 28);
    net->t_conv1_out = create_tensor(batch_size, 8, 26, 26);
    net->t_relu_out  = create_tensor(batch_size, 8, 26, 26);
    net->t_pool_out  = create_tensor(batch_size, 8, 13, 13);
    net->t_fc_out    = create_tensor(batch_size, 1, 1, 10); // Treating FC out as 1x1x10 or just flat 10

    // --- 3. Allocate Gradient Buffers ---
    net->t_fc_grad    = create_tensor(batch_size, 1, 1, 10);
    net->t_pool_grad  = create_tensor(batch_size, 8, 13, 13);
    net->t_relu_grad  = create_tensor(batch_size, 8, 26, 26);
    net->t_conv1_grad = create_tensor(batch_size, 8, 26, 26);

    return net;
}

// ------------------------------------------------------------------
// FORWARD & BACKWARD PIPELINE
// ------------------------------------------------------------------
float forward_backward_pass(Network* net, Tensor* input_batch, unsigned char* labels, float learning_rate) {
    
    memcpy(net->input->data, input_batch->data, net->input->total_size * sizeof(float));

    conv2d_forward(net->conv1, net->input, net->t_conv1_out);
    relu_forward(net->t_conv1_out, net->t_relu_out);
    max_pool_forward(net->t_relu_out, net->t_pool_out, 2, 2); // Kernel=2, Stride=2
    fc_forward(net->fc1, net->t_pool_out, net->t_fc_out);

    float loss = softmax_loss_backward(net->t_fc_out, labels, net->t_fc_grad);

    fc_backward(net->fc1, net->t_pool_out, net->t_fc_grad, net->t_pool_grad);
    
    max_pool_backward(net->t_relu_out, net->t_pool_grad, net->t_relu_grad, 2, 2);
    
    relu_backward(net->t_conv1_out, net->t_relu_grad, net->t_conv1_grad);
    
    conv2d_backward(net->conv1, net->input, net->t_conv1_grad, NULL); 

    return loss;
}

// ------------------------------------------------------------------
// OPTIMIZER (SGD)
// ------------------------------------------------------------------
// In Distributed training, this is called AFTER MPI_Allreduce
void optimizer_step(Network* net, float lr) {
    // Update Conv1
    #pragma omp parallel for
    for (int i = 0; i < net->conv1->weights->total_size; i++) {
        net->conv1->weights->data[i] -= lr * net->conv1->grad_weights->data[i];
    }
    #pragma omp parallel for
    for (int i = 0; i < net->conv1->biases->total_size; i++) {
        net->conv1->biases->data[i] -= lr * net->conv1->grad_biases->data[i];
    }

    // Update FC1
    #pragma omp parallel for
    for (int i = 0; i < net->fc1->weights->total_size; i++) {
        net->fc1->weights->data[i] -= lr * net->fc1->grad_weights->data[i];
    }
    #pragma omp parallel for
    for (int i = 0; i < net->fc1->biases->total_size; i++) {
        net->fc1->biases->data[i] -= lr * net->fc1->grad_biases->data[i];
    }
}

void free_network(Network* net) {
    // Free intermediate tensors
    free_tensor(net->input);
    free_tensor(net->t_conv1_out);
    free_tensor(net->t_relu_out);
    free_tensor(net->t_pool_out);
    free_tensor(net->t_fc_out);
    free_tensor(net->t_fc_grad);
    free_tensor(net->t_pool_grad);
    free_tensor(net->t_relu_grad);
    free_tensor(net->t_conv1_grad);

    // Free layers (You'll need to implement free_conv_layer etc in their files)
    // free(net->conv1); 
    // free(net->fc1);
    free(net);
}

int argmax(float* data, int size) {
    int max_idx = 0;
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int evaluate_accuracy(Network* net, Tensor* input_data, unsigned char* labels, int count) {
    int correct = 0;
    int batch_size = net->input->b; // Should match what the net was built with
    
    // Create temporary tensor wrappers
    Tensor* batch_input = create_tensor(batch_size, 1, 28, 28);
    
    // Loop through data in chunks of 'batch_size'
    for (int i = 0; i < count; i += batch_size) {
        int current_batch = (i + batch_size > count) ? (count - i) : batch_size;
        
        // 1. Load Batch
        // Note: For simplicity in this eval function, we copy row by row
        // In production, you'd handle the last partial batch carefully.
        // Here we assume dataset size is divisible or we just drop the last few for speed.
        if (current_batch < batch_size) break; 

        int img_size = 28 * 28;
        #pragma omp parallel for
        for (int k = 0; k < batch_size * img_size; k++) {
            batch_input->data[k] = input_data->data[(i * img_size) + k];
        }

        // 2. Forward Pass Only (No Backward, No Loss Calc)
        conv2d_forward(net->conv1, batch_input, net->t_conv1_out);
        relu_forward(net->t_conv1_out, net->t_relu_out);
        max_pool_forward(net->t_relu_out, net->t_pool_out, 2, 2);
        fc_forward(net->fc1, net->t_pool_out, net->t_fc_out);

        // 3. Check Predictions
        for (int b = 0; b < batch_size; b++) {
            // Logits for this image are at: net->t_fc_out->data[b * 10]
            int predicted = argmax(&net->t_fc_out->data[b * 10], 10);
            int actual = (int)labels[i + b];
            if (predicted == actual) {
                correct++;
            }
        }
    }
    
    free_tensor(batch_input);
    return correct;
}
