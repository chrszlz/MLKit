//
//  NeuralNetwork.swift
//  MLKit
//
//  Created by Kesav Mulakaluri on 12/27/15.
//  Copyright © 2015 Kesav Mulakaluri. All rights reserved.
//

import Foundation

/// Represents a feed forward neural network.
public protocol NeuralNetwork: MLBlock {
    
    /// The toplogy of the network. Specifies the series of transforms to apply
    /// to the input.
    var topology: [MLBlock] { get set }
}