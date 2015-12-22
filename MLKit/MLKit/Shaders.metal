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