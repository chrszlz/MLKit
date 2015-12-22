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
        guard let libraryPath = bundle.URLForResource("default", withExtension: "metallib", subdirectory: "Versions/A/Resources") else {
            fatalError("Unable to find default metallib")
        }
        
        guard let library = try? device.newLibraryWithFile(libraryPath.path!) else {
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
        // Get the shader and configure the command encoder.
        guard let scalingFunction = library.newFunctionWithName("matrix_scale") else {
            fatalError("No shader named matrix_scale.")
        }
        
        guard let pipeline = try? device.newComputePipelineStateWithFunction(scalingFunction) else {
            fatalError("Could not create compute pipeline with the matrix_scale shader.")
        }
        
        let commandBuffer = commandQueue.commandBuffer()
        let commandEncoder = commandBuffer.computeCommandEncoder()
        commandEncoder.setComputePipelineState(pipeline)
        
        // Load the data into MTLBuffers the shader can access.
        var scalingFactor = c
        var input = a.elements
        var output = [Float](count: a.elements.count, repeatedValue: 0)
        let size = a.elements.count * sizeof(Float)
        
        let inputBuffer = device.newBufferWithBytes(&input, length: size, options: .StorageModePrivate)
        let outputBuffer = device.newBufferWithBytes(&output, length: size, options: .StorageModePrivate)
        commandEncoder.setBytes(&scalingFactor, length: sizeof(Float), atIndex: 0)
        commandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 1)
        commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 2)
        
        // Set the number of threads to be executed in parallel.
        let threadsPerGroup = MTLSize(width: 16, height: 1, depth: 1)
        let numThreadGroups = MTLSize(width: (input.count+15)/threadsPerGroup.width, height: 1, depth: 1)
        commandEncoder.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
        
        // Commit the computations to the GPU and wait for it to finish.
        commandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Grab the data from the from the MTLBuffer and return the new scaled
        // matrix.
        let data = NSData(bytesNoCopy: outputBuffer.contents(), length: size, freeWhenDone: false)
        var result = [Float](count: output.count, repeatedValue: 0)
        data.getBytes(&result, length: size)
        
        return Matrix(rows: a.rows, columns: a.columns, elements: result)
    }
}