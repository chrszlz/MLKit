//
//  MLBlock.swift
//  MLKit
//
//  Created by Kesav Mulakaluri on 12/27/15.
//  Copyright © 2015 Kesav Mulakaluri. All rights reserved.
//

import Foundation

/// An `MLBlock` is a basic unit of functionality in `MLKit`. It is responsible
/// for taking some input, transforming the data, and providing some output.
public protocol MLBlock {
    
    /// The name of the `MLBlock`.
    var name: String { get }
    
    /// Returns a `Matrix` by applying some transform to `input`.
    func apply(_ input: Matrix) -> Matrix
}
