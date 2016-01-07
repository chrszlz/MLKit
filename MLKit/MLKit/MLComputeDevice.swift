//
//  MLComputeDevice.swift
//  MLKit
//
//  Created by Kesav Mulakaluri on 12/22/15.
//  Copyright © 2015 Kesav Mulakaluri. All rights reserved.
//

import Foundation

public protocol MLComputeDevice {
    
    // MARK: - Matrix Operations
    
    /// Returns `a` + `b`.
    func addMatrices(a a: Matrix, b: Matrix) -> Matrix
    
    /// Returns `a` - `b`.
    func subtractMatrices(a a: Matrix, b: Matrix) -> Matrix
    
    /// Returns `a` * `b`.
    func multiplyMatrices(a a: Matrix, b: Matrix) -> Matrix
    
    /// Multiplies each element in `a` by `c`.
    func scaleMatrix(a: Matrix, by c: Float) ->  Matrix
    
    
    // MARK: - Activations
    
    /// Applies the sigmoid function to each element in `a`.
    func applySigmoid(a: Matrix) -> Matrix
    
    /// Applies the derivative of the sigmoid function to each element in `a`.
    func applySigmoidDerivative(a: Matrix) -> Matrix
    
    /// Applies the hyperbolic tangent to each element in `a`.
    func applyTanh(a: Matrix) -> Matrix
    
    /// Applies the Rectified Linear activation to each element in `a`.
    func applyRelu(a: Matrix) -> Matrix
}