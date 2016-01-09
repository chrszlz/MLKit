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
    public let name: String
    
    /// The weights for this layer.
    public var weights: Matrix
    
    /// The bias for this layer.
    public var bias: Matrix
    
    
    // MARK: - Initializers
    
    /// Initializes an InnerProduct Layer.
    ///
    /// - parameter name: The name of the layer.
    /// - parameter weights: The weights for the layer.
    /// - parameter bias: The bias for the layer.
    public init(name: String, weights: Matrix, bias: Matrix) {
        self.name = name
        self.weights = weights
        self.bias = bias
    }
    
    /// Returns the weights * input + bias
    public func apply(input: Matrix) -> Matrix {
        return (weights * input) + bias
    }
}