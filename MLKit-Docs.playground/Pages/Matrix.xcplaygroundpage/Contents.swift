//: [Previous](@previous)

import Foundation
import MLKit

// Real docs coming soon. Just using this as a way to test the matrix 
// because fuck unit tests...

// Basic matrix test
let m = Matrix(elements: [[1,2], [3,4], [5,6], [7,8]])
print(m)

// Matrix copy test
let n = m.copy()
m == n

// Matrix multiplication test
print(3.0 * m)

// Matrix subtraction test
print(m - n)

// Matrix addition test
print(m + n)
//: [Next](@next)
