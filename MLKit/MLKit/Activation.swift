//
//  Activation.swift
//  MLKit
//
//  Created by Kesav Mulakaluri on 12/27/15.
//  Copyright © 2015 Kesav Mulakaluri. All rights reserved.
//

import Foundation

/// Represents an activation function used in Neural Networks.
public protocol Activation: MLBlock {
    
    /// Applies the derivative of the activation function to `input`.
    func applyDerivative(_ input: Matrix) -> Matrix
}
