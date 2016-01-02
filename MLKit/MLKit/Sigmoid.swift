//
//  Sigmoid.swift
//  MLKit
//
//  Created by Kesav Mulakaluri on 1/1/16.
//  Copyright © 2016 Kesav Mulakaluri. All rights reserved.
//

import Foundation

struct Sigmoid: Activation {
    
    /// The name of the sigmoid activation block.
    var name: String
    
    /// Initializes a Sigmoid MLBlock with name `name`.
    init(name: String) {
        self.name = name
    }
    
    /// Applies the sigmoid function to `input`.
    func apply(input: Matrix) -> Matrix {
        return MLComputeOptions.computeDevice.applySigmoid(input)
    }
    
    /// Applies the derivative of the sigmoid function to `input`.
    func applyDerivative(input: Matrix) -> Matrix {
        return MLComputeOptions.computeDevice.applySigmoid(input)
    }
}