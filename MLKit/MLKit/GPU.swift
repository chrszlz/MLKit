//
//  GPU.swift
//  MLKit
//
//  Created by Kesav Mulakaluri on 12/19/15.
//  Copyright © 2015 Kesav Mulakaluri. All rights reserved.
//

import Foundation
import Metal

/// GPU is a light weight wrapper around Metal to perform efficient
/// matrix computations.
class GPU {
    
    /// The shared instance used to get access to the GPU.
    static let deviceGPU = GPU()
    
    /// The abstraction for the GPU.
    private let device: MTLDevice
    
    /// A collection of all the shaders in MLKit.
    private let library: MTLLibrary
    
    /// The queue that contains all the commands that need to be executed on
    /// the GPU.
    private let commandQueue: MTLCommandQueue
    
    
    // MARK: - Initializers
    
    init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Unable to create MTLDevice.")
        }
        
        let bundle = NSBundle(forClass: GPU.self)
        guard let shadersSource = bundle.pathForResource("Shaders", ofType: "metal") else {
            fatalError("Unable to find shaders.")
        }
        
        guard let library = try? device.newLibraryWithSource(shadersSource, options: nil) else {
            fatalError("Unable to create MTLLibrary.")
        }
        
        self.device = device
        self.library = library
        self.commandQueue = device.newCommandQueue()
    }

    
    // MARK: - Matrix Operations
    
    /// Returns `a` + `b`.
    func addMatrices(a: Matrix, b: Matrix) -> Matrix {
        return a
    }
    
    /// Returns `a` - `b`.
    func subtractMatrices(a: Matrix, b: Matrix) -> Matrix {
        return a
    }
    
    /// Returns `a` * `b`.
    func multiplyMatrices(a: Matrix, b: Matrix) -> Matrix {
        return a
    }
    
    /// Multiplies each element in `a` by `c`.
    func scaleMatrix(a: Matrix, by c: Float) ->  Matrix {
        return a
    }
}