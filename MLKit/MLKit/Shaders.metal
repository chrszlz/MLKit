//
//  Shaders.metal
//  MLKit
//
//  Created by Kesav Mulakaluri on 12/19/15.
//  Copyright © 2015 Kesav Mulakaluri. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

// Multiplies each element in a matrix by some scalar.
kernel void matrix_scale(constant float &scalingFactor [[buffer(0)]],
                        const device float *input [[buffer(1)]],
                        device float *output [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    output[gid] = scalingFactor * input[gid];
}

// Adds two matrices together.
kernel void matrix_add(const device float *input1 [[ buffer(0) ]],
                       const device float *input2 [[ buffer(1) ]],
                       device float *output [[ buffer(2) ]],
                       uint gid [[ thread_position_in_grid ]]) {
    output[gid] = input1[gid] + input2[gid];
}

// Subtracts two matrices.
kernel void matrix_subtract(const device float *input1 [[ buffer(0) ]],
                            const device float *input2 [[ buffer(1) ]],
                            device float *output [[ buffer(2) ]],
                            uint gid [[ thread_position_in_grid ]]) {
    output[gid] = input1[gid] - input2[gid];
}

// Applies the sigmoid function to each element in input.
kernel void activation_sigmoid(const device float *input [[ buffer(0) ]],
                    device float *output [[ buffer(1) ]],
                    uint gid [[ thread_position_in_grid ]]) {
    output[gid] = 1.0/(1.0 + exp(-1.0 * input[gid]));
}

// Applies the derivative of the sigmoid function to each element in input.
kernel void activation_sigmoid_derivative(const device float *input [[ buffer(0) ]],
                               device float *output [[ buffer(1) ]],
                               uint gid [[ thread_position_in_grid ]]) {
    float s = 1.0/(1.0 + exp(-1.0 * input[gid]));
    output[gid] = s * (1 - s);
}

// Applies the tanh function to each element in input.
kernel void activation_tanh(const device float *input [[ buffer(0) ]],
                               device float *output [[ buffer(1) ]],
                               uint gid [[ thread_position_in_grid ]]) {
    output[gid] = tanh(input[gid]);
}

// Applies the derivative of the tanh function to each element in input.
kernel void activation_tanh_derivative(const device float *input [[ buffer(0) ]],
                            device float *output [[ buffer(1) ]],
                            uint gid [[ thread_position_in_grid ]]) {
    float res = tanh(input[gid]);
    output[gid] = 1.0 - pow(res, 2.0);
}

// Applies the relu activation function to each element in input.
kernel void activation_relu(const device float *input [[ buffer(0) ]],
                            device float *output [[ buffer(1) ]],
                            uint gid [[ thread_position_in_grid ]]) {
    output[gid] = max(0.0, input[gid]);
}