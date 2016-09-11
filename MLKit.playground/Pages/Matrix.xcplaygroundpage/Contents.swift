//: [Previous](@previous)

import Foundation
import MLKit

// Real docs coming soon. Just using this as a way to test the matrix 
// because fuck unit tests...

let r = Matrix(rows: 2, columns: 2, policy: .xavier)

let m = Matrix([[1,2],[3,4]])
let res = m.T

let res2 = 3.0 * m
let res3 = m + m
let res4 = (2.0 * m) - m
//: [Next](@next)
