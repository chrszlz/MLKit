//
//  MLComputeOptions.swift
//  MLKit
//
//  Created by Kesav Mulakaluri on 12/22/15.
//  Copyright © 2015 Kesav Mulakaluri. All rights reserved.
//

import Foundation

public enum MLComputeMode {
    case CPU
    case GPU
}

public struct MLComputeOptions {
    
    public static var computeMode: MLComputeMode = .CPU
    
    public static var computeDevice: MLComputeDevice {
        switch computeMode {
        case .CPU:
            return CPU()
        case .GPU:
            return GPU.deviceGPU
        }
    }
}


// MARK: - Convenience Functions

public func MLSetComputeMode(computeMode: MLComputeMode) {
    MLComputeOptions.computeMode = computeMode
}

public func MLGetComputeMode() -> MLComputeMode {
    return MLComputeOptions.computeMode
}

public func MLGetComputeDevice() -> MLComputeDevice {
    return MLComputeOptions.computeDevice
}