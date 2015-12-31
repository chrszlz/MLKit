//
//  InnerProduct.swift
//  MLKit
//
//  Created by Kesav Mulakaluri on 12/28/15.
//  Copyright © 2015 Kesav Mulakaluri. All rights reserved.
//

import Foundation

public struct InnerProduct: Layer {
    
    
    // MARK: - MLBlock
    
    public var name: String
    
    public var input: Matrix
    
    public var output: Matrix {
        return (weights * input) + bias
    }
    
    
    // MARK: - Layer
    
    public var weights: Matrix
    
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
        self.input = Matrix.zeros(rows: 1, columns: 1)
        self.weights = Matrix(shape: weightsShape, policy: policy)
        self.bias = Matrix(shape: biasShape, policy: policy)
    }
}