//
//  InnerProduct.swift
//  MLKit
//
//  Created by Kesav Mulakaluri on 12/28/15.
//  Copyright © 2015 Kesav Mulakaluri. All rights reserved.
//

import Foundation

public struct InnerProduct: Layer {
    
    /// The name of the layer.
    public var name: String
    
    /// The weights for this layer.
    public var weights: Matrix
    
    /// The bias for this layer.
    public var bias: Matrix
    
    
    // MARK: - Initializers
    
    /// Initializes the InnerProduct layer.
    ///
    /// - parameter name: The name of the layer.
    /// - parameter weightsShape: The shape of the weights matrix.
    /// - parameter weightInitialization: How to initialize the weights and bias.
    /// - parameter biasShape: The shape of the bias matrix.
    public init(name: String, weightsShape: Shape, weightInitialization policy: InitializationPolicy, biasShape: Shape) {
        self.name = name
        self.weights = Matrix(shape: weightsShape, policy: policy)
        self.bias = Matrix(shape: biasShape, policy: policy)
    }
    
    /// Returns the weights * input + bias
    public func apply(input: Matrix) -> Matrix {
        return (weights * input) + bias
    }
}