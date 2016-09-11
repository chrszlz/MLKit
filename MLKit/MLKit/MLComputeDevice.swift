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
    func addMatrices(a: Matrix, b: Matrix) -> Matrix
    
    /// Returns `a` - `b`.
    func subtractMatrices(a: Matrix, b: Matrix) -> Matrix
    
    /// Returns `a` * `b`.
    func multiplyMatrices(a: Matrix, b: Matrix) -> Matrix
    
    /// Multiplies each element in `a` by `c`.
    func scaleMatrix(_ a: Matrix, by c: Float) ->  Matrix
    
    /// Returns the sum of all the elements in `a`.
    func sumMatrix(_ a: Matrix) -> Float
    
    /// Exponentiates each element in `a`.
    func expMatrix(_ a: Matrix) -> Matrix
    
    // MARK: - Activation Functions
    
    /// Applies the sigmoid function to each element in `a`.
    func applySigmoid(_ a: Matrix) -> Matrix
    
    /// Applies the derivative of the sigmoid function to each element in `a`.
    func applySigmoidDerivative(_ a: Matrix) -> Matrix
    
    /// Applies the hyperbolic tangent to each element in `a`.
    func applyTanh(_ a: Matrix) -> Matrix
    
    /// Applies the derivative of the hyperbolic tangent to each element in `a`.
    func applyTanhDerivative(_ a: Matrix) -> Matrix
    
    /// Applies the Rectified Linear activation to each element in `a`.
    func applyRelu(_ a: Matrix) -> Matrix
    
    /// Applies the Rectified Linear activation derivative to each element in `a`.
    func applyReluDerivative(_ a: Matrix) -> Matrix
}
