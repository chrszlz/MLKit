//
//  Softmax.swift
//  MLKit
//
//  Created by Kesav Mulakaluri on 1/8/16.
//  Copyright © 2016 Kesav Mulakaluri. All rights reserved.
//

import Foundation

struct Softmax: MLBlock {
    
    /// The name of the Softmax block.
    let name: String
    
    /// Initializes a Softmax block with name `name`.
    init(name: String) {
        self.name = name
    }
    
    /// Applies the softmax function to `input`.
    func apply(input: Matrix) -> Matrix {
        return input
    }
}