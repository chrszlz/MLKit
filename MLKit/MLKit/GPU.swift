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
        let pipelineState = initShader(function: "matrix_scale")
        
        let commandBuffer = commandQueue.commandBuffer()
        let commandEncoder = commandBuffer.computeCommandEncoder()
        
        // Load the data into MTLBuffers the shader can access.
        var scale = c
        
        let input = a.elements
        let inputBuffer = newBuffer(input)
        let size = input.count * sizeof(Float)
        
        var output = [Float](count: input.count, repeatedValue: 1.0)
        let outputBuffer = newBuffer(output)
        
        // Set encoder inputs/output
        commandEncoder.setBytes(&scale, length: sizeof(Float), atIndex: 0)
        commandEncoder.setBuffer(inputBuffer, offset: 0, atIndex: 1)
        commandEncoder.setBuffer(outputBuffer, offset: 0, atIndex: 2)
        
        // Set fader function
        commandEncoder.setComputePipelineState(pipelineState)
        
        // Set the number of threads to be executed in parallel.
        let threadsPerGroup = MTLSize(width: pipelineState.threadExecutionWidth, height: 1, depth: 1)
        let numThreadGroups = MTLSize(width: input.count/threadsPerGroup.width, height: 1, depth: 1)
        commandEncoder.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
        
        // Commit the computations to the GPU and wait for it to finish.
        commandEncoder.endEncoding()

        // Run it - Do we need this?
        commandBuffer.enqueue()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Grab the data from the from the MTLBuffer and return the new scaled
        // matrix.
        let data = NSData(bytesNoCopy: outputBuffer.contents(), length: size, freeWhenDone: false)
        data.getBytes(&output, length: size)
        
        return Matrix(rows: a.rows, columns: a.columns, elements: output)
    }
    
    
    // MARK - Utility Functions
    
    private func initShader(function function: String) -> MTLComputePipelineState {
        // Get the shader and configure the command encoder.
        guard let scalingFunction = library.newFunctionWithName(function) else {
            fatalError("No shader named \(function).")
        }
        
        // Get the pipeline state
        guard let pipeline = try? device.newComputePipelineStateWithFunction(scalingFunction) else {
            fatalError("Could not create compute pipeline with the \(function) shader.")
        }
        
        return pipeline
    }
    
    private func newBuffer<T>(var vector: [T]) -> MTLBuffer {
        let size = vector.count * sizeof(T)
        return device.newBufferWithBytes(&vector, length: size, options: .CPUCacheModeDefaultCache)
    }

//    @objc private func newBuffer(var vector: [Float]) -> MTLBuffer {
//        let size = vector.count * sizeof(Float)
//        return device.newBufferWithBytes(&vector, length: size, options: .StorageModePrivate)
//    }
    
}