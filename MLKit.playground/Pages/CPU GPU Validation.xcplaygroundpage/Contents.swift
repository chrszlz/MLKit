//: [Previous](@previous)

import Foundation
import MLKit

// Shared variables
let m: Matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

// CPU Mode
MLSetComputeMode(.CPU)

let addCPU  = m + m
let subCPU  = m - m
let multCPU = m * 3

// GPU Mode
MLSetComputeMode(.GPU)

let addGPU  = m + m
let subGPU  = m - m
let multGPU = 3 * m

// Validate
// If all evaluate to true, CPU & GPU calculations are consistent
addCPU  == addGPU
subCPU  == subGPU
multCPU == multGPU
//: [Next](@next)
