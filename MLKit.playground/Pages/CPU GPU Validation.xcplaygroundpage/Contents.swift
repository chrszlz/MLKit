//: [Previous](@previous)

import Foundation
import MLKit

// Shared variables
let m: Matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

// CPU Mode
MLSetComputeMode(.CPU)

let addCPU  = m + m
print(addCPU)
let subCPU  = m - m
print(subCPU)
let multCPU = m * 3
print(multCPU)
print(m)
let vecMultCPU = m * m
print(vecMultCPU)

// GPU Mode
MLSetComputeMode(.GPU)

let addGPU  = m + m
print(addGPU)
let subGPU  = m - m
print(subGPU)
let multGPU = 3 * m
print(multGPU)

// Validate
// If all evaluate to true, CPU & GPU calculations are consistent
addCPU  == addGPU
subCPU  == subGPU
multCPU == multGPU
//: [Next](@next)
