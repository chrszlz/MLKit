//
//  ReLU.swift
//  MLKit
//
//  Created by Kesav Mulakaluri on 1/4/16.
//  Copyright © 2016 Kesav Mulakaluri. All rights reserved.
//

import Foundation

struct ReLU: Activation {
    
    /// The name of the ReLU block.
    let name: String
    
    /// Initializes a ReLU activation block with name `name`.
    init(name: String) {
        self.name = name
    }
    
    /// Applies the ReLU activation function to each element in `input`.
    func apply(_ input: Matrix) -> Matrix {
        return MLComputeOptions.computeDevice.applyRelu(input)
    }
    
    /// Applies the ReLU activation function derivative to each element in `input`.
    func applyDerivative(_ input: Matrix) -> Matrix {
        return MLComputeOptions.computeDevice.applyReluDerivative(input)
    }
}
