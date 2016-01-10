//
//  Matrix.swift
//  MLKit
//
//  Created by Kesav Mulakaluri on 12/16/15.
//  Copyright © 2015 Kesav Mulakaluri. All rights reserved.
//

import Foundation
import Accelerate

/// The method used to initialize the elements in the matrix.
public enum InitializationPolicy {
    
    /// Randomly initialize each element with number between min and max.
    case Random(min: Float, max: Float)
    
    /// Initialize the matrix using Xavier initialization.
    case Xavier
}

public struct Matrix: Equatable {
    
    /// The number of rows in this matrix.
    public let rows: Int
    
    /// The number of columns in this matrix.
    public let columns: Int
    
    /// The shape of this matrix.
    public let shape: Shape
    
    /// The transpose of this matrix.
    public var T: Matrix {
        return self.transpose()
    }
    
    /// Returns the element in the matrix at row `row` and column `column`.
    public subscript(row: Int, column: Int) -> Float {
        get {
            guard row >= 0 && row < self.rows else {
                fatalError("Invalid row: \(row)")
            }
            
            guard column >= 0 && column < self.columns else {
                fatalError("Invalid column: \(column)")
            }
            
            return self.elements[row * self.columns + column]
        }
    }
    
    /// The backing store for this matrix.
    var elements: [Float]
    

    // MARK: - Initializers
    
    /// Initializes a `rows` by `columns` matrix using the specified `policy`.
    ///
    /// - parameter rows: The number of rows in the matrix.
    /// - parameter columns: The number of columns in the matrix.
    /// - parameter policy: The `InitializationPolicy` used to initialize each
    ///                     element in the matrix.
    public init(rows: Int, columns: Int, policy: InitializationPolicy) {
        let rows = rows
        let columns = columns
        let count = rows * columns
        
        var elements = [Float](count: count, repeatedValue: 0.0)
        for i in 0..<count {
            switch policy {
            case .Random(let min, let max):
                elements[i] = Float.random(min, max)
            case .Xavier:
                elements[i] = Float.random()/sqrt(Float(columns))
            }
        }
        
        self.init(rows: rows, columns: columns, elements: elements)
    }
    
    /// Initializes a `rows` by `columns` matrix using the specified `policy`.
    ///
    /// - parameter shape: The number of rows and columns in the matrix.
    /// - parameter policy: The `InitializationPolicy` used to initialize each
    ///                     element in the matrix.
    public init(shape: Shape, policy: InitializationPolicy) {
        self.init(rows: shape.rows, columns: shape.columns, policy: policy)
    }
    
    /// Initializes the matrix from a 2D array.
    ///
    /// - parameter elements: The 2D array used to initialize the matrix.
    public init(_ elements: [[Float]]) {
        let rows = elements.count
        let columns = elements[0].count
        let elements = elements.flatMap { $0 }
        
        self.init(rows: rows, columns: columns, elements: elements)
    }
    
    /// Returns a `rows` by `columns` matrix of zeros.
    ///
    /// - parameter rows: The number of rows in the matrix.
    /// - parameter: columns: The number of columns in the matrix.
    public static func zeros(rows rows: Int, columns: Int) -> Matrix {
        let elements = [Float](count: rows * columns, repeatedValue: 0)
        return Matrix(rows: rows, columns: columns, elements: elements)
    }
    
    /// Returns a `rows` by `columns` matrix of zeros.
    ///
    /// - parameter shape: The number of rows and columns in the matrix.
    public static func zeros(shape: Shape) -> Matrix {
        return self.zeros(rows: shape.rows, columns: shape.columns)
    }
    
    /// Returns a `rows` by `columns` matrix of ones.
    ///
    /// - parameter rows: The number of rows in the matrix.
    /// - parameter columns: The number of columns in the matrix.
    public static func ones(rows rows: Int, columns: Int) -> Matrix {
        let elements = [Float](count: rows * columns, repeatedValue: 1)
        return Matrix(rows: rows, columns: columns, elements: elements)
    }
    
    /// Returns a `rows` by `columns` matrix of ones.
    ///
    /// - parameter shape: The number of rows and columns in the matrix.
    public static func ones(shape: Shape) -> Matrix {
        return self.ones(rows: shape.rows, columns: shape.columns)
    }
    
    /// The designated initializer. Creates a `rows` by `columns` matrix with
    /// `elements`.
    init(rows: Int, columns: Int, elements: [Float]) {
        guard rows > 0 && columns > 0 else {
            fatalError("Invalid dimensions. Matrix must be at least 1x1.")
        }
        
        self.rows = rows
        self.columns = columns
        self.elements = elements
        self.shape = Shape(rows, columns)
    }
    
    
    // MARK: - Operations
    
    /// Returns the sum of all the elements in this matrix.
    public func sum() -> Float {
        // TODO: Change this to MLComputeDevice once implemented in GPU.
        return CPU().sumMatrix(self)
    }
    
    public func exp() -> Matrix {
        return MLComputeOptions.computeDevice.expMatrix(self)
    }
    
    /// Returns the transpose of this matrix.
    private func transpose() -> Matrix {
        var res = Matrix.zeros(rows: self.columns, columns: self.rows)
        vDSP_mtrans(self.elements, 1, &(res.elements), 1, vDSP_Length(res.rows), vDSP_Length(res.columns))
        return res
    }
    
    /// Returns the sum of the two matrices (a + b).
    private static func add(a: Matrix, b: Matrix) -> Matrix {
        guard (a.shape == b.shape) else {
            fatalError("Shape of a(\(a.shape) must be equal to shape of b(\(b.shape) for addition.")
        }
        
        return MLComputeOptions.computeDevice.addMatrices(a: a, b: b)
    }
    
    /// Returns the difference of the two matrices (a - b).
    private static func subtract(a: Matrix, b: Matrix) -> Matrix {
        guard (a.shape == b.shape) else {
            fatalError("Shape of a(\(a.shape) must be equal to shape of b(\(b.shape) for subtraction.")
        }
        
        return MLComputeOptions.computeDevice.subtractMatrices(a: a, b: b)
    }
    
    /// Returns the product of two matrices (a * b).
    private static func multiply(a: Matrix, b: Matrix) -> Matrix {
        guard (a.columns == b.rows) else {
            fatalError("Number of columns in a(\(a.columns) must be equal to number of rows in b(\(b.rows) for multiplication.")
        }
        
        return MLComputeOptions.computeDevice.multiplyMatrices(a: a, b: b)
    }
    
    /// Returns a `Matrix` where each element in `a` is multiplied by `s`.
    private static func scale(s: Float, a: Matrix) -> Matrix {
        return MLComputeOptions.computeDevice.scaleMatrix(a, by: s)
    }
}

// MARK: - Operators

public func + (lhs: Matrix, rhs: Matrix) -> Matrix {
    return Matrix.add(lhs, b: rhs)
}

public func - (lhs: Matrix, rhs: Matrix) -> Matrix {
    return Matrix.subtract(lhs, b: rhs)
}

public func * (lhs: Matrix, rhs: Matrix) -> Matrix {
    return Matrix.multiply(lhs, b: rhs)
}

public func * (lhs: Float, rhs: Matrix) -> Matrix {
    return Matrix.scale(lhs, a: rhs)
}

public func * (lhs: Matrix, rhs: Float) -> Matrix {
    return Matrix.scale(rhs, a: lhs)
}


// MARK: - Equatable

public func == (lhs: Matrix, rhs: Matrix) -> Bool {
    return lhs.elements == rhs.elements
}


// MARK: - ArrayLiteralConvertible

extension Matrix: ArrayLiteralConvertible {
    public init(arrayLiteral elements: [Float]...) {
        self = Matrix(elements)
    }
}


// MARK: - CustomStringConvertible

extension Matrix: CustomStringConvertible {
    public var description: String {
        var output = "\t\(self.rows)x\(self.columns) : \(self.elements[0].dynamicType) : \(MLComputeOptions.computeMode)\n\n"
 
        // Calculates how many indices to print within each axis.
        func determineAxisIndices(axisLength axisLength: Int, axisLimit: Int) -> [Int] {
            guard (axisLength > 2*axisLimit) else {
                return [Int](0..<axisLength)
            }
            
            var indices = [Int] ( 0..<axisLimit ) // Beginning indices
            indices += [Int] ( (axisLength-1)-(axisLimit-1)..<axisLength ) // End indices
            indices.insert(-1, atIndex: axisLimit) // Inserts -1 to mark ...
            
            return indices
        }
        
        // Determine row and column indices to print.
        let LIMITS = Shape(3, 3)
        let rowIndices = determineAxisIndices(axisLength: self.rows, axisLimit: LIMITS.rows)
        let colIndices = determineAxisIndices(axisLength: self.columns, axisLimit: LIMITS.columns)
        
        for row in rowIndices {
            guard (row >= 0) else {
                output += "\t• • •\n"
                continue
            }
            
            for col in colIndices {
                guard (col >= 0) else {
                    output += "    • • •"
                    continue
                }
                output += "\t\(NSString(format:"% 1.2f", self[row, col]))"
            }
            output += "\n"
        }
        
        return output
    }
}


// MARK: - Float Extensions

extension Float {
    
    /// Returns a float between `min` and `max`. If no arguments are passed in,
    /// `min` defaults to 0 and `max` defaults to 100.
    public static func random(min: Float = 0, _ max: Float = 100) -> Float {
        return (Float(arc4random()) / 0xFFFFFFFF) * (max - min) + min
    }
}
