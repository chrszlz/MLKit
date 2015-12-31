//
//  Layer.swift
//  MLKit
//
//  Created by Kesav Mulakaluri on 12/27/15.
//  Copyright © 2015 Kesav Mulakaluri. All rights reserved.
//

import Foundation

/// Represents a layer of a Neural Network.
public protocol Layer: MLBlock {
    
    /// The weights for this layer.
    var weights: Matrix { get set }
    
    /// The bias for this layer.
    var bias: Matrix { get set }
}