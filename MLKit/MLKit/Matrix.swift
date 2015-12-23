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
///
/// - Random(min, max): Randomly initialize each element with a
///                     number between min and max.
/// - Xavier: Initalize the elements using Xavier initialization.
public enum InitializationPolicy {
    case Random(Float, Float)
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
        let elements = [Float(0)]
        
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
    /// - parameter rows: The number of rows in the matrix.
    /// - parameter: columns: The number of columns in the matrix.
    public static func zeros(rows rows: Int, columns: Int) -> Matrix {
        let elements = [Float](count: rows * columns, repeatedValue: 0)
        return Matrix(rows: rows, columns: columns, elements: elements)
    }
    
    /// Returns a `rows` by `columns` matrix of zeros.
    /// - parameter shape: The number of rows and columns in the matrix.
    public static func zeros(shape: Shape) -> Matrix {
        return self.zeros(rows: shape.rows, columns: shape.columns)
    }
    
    /// Returns a `rows` by `columns` matrix of ones.
    /// - parameter rows: The number of rows in the matrix.
    /// - parameter columns: The number of columns in the matrix.
    public static func ones(rows rows: Int, columns: Int) -> Matrix {
        let elements = [Float](count: rows * columns, repeatedValue: 1)
        return Matrix(rows: rows, columns: columns, elements: elements)
    }
    
    /// Returns a `rows` by `columns` matrix of ones.
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
    
    /// Returns the transpose of this matrix.
    private func transpose() -> Matrix {
        var res = Matrix.zeros(rows: self.columns, columns: self.rows)
        vDSP_mtrans(self.elements, 1, &(res.elements), 1, vDSP_Length(res.rows), vDSP_Length(res.columns))
        return res
    }
    
    /// Returns the sum of the two matrices (a + b).
    private static func add(a: Matrix, b: Matrix) -> Matrix {
        return MLComputeOptions.computeDevice.addMatrices(a: a, b: b)
    }
    
    /// Returns the difference of the two matrices (a - b).
    private static func subtract(a: Matrix, b: Matrix) -> Matrix {
        return MLComputeOptions.computeDevice.subtractMatrices(a: a, b: b)
    }
    
    /// Returns the product of two matrices (a * b).
    private static func multiply(a: Matrix, b: Matrix) -> Matrix {
        return a
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

// MARK: Literal

extension Matrix: ArrayLiteralConvertible {
    public init(arrayLiteral elements: [Float]...) {
        let data = elements
        let matrix = Matrix(data)
        self = matrix
    }
}

// MARK: - CustomStringConvertible

extension Matrix: CustomStringConvertible {
    public var description: String {
        var output = "\t\(self.rows)x\(self.columns) : \(self.elements[0].dynamicType)\n\n"
        
        let WIDE = 3
        let TALL = 3
        
        var rowIndices: [Int] = []
        var colIndices: [Int] = []
        
        // Determine row indexes
        if (self.rows > 2*TALL) {
            for r in (0..<TALL) { rowIndices += [r] }
            rowIndices += [-1]
            for r in (self.rows-TALL..<self.rows) { rowIndices += [r] }
        } else {
            for r in (0..<self.rows) { rowIndices += [r] }
        }
        
        // Determine column indexes
        if (self.columns > 2*WIDE) {
            for c in (0..<WIDE) { colIndices += [c] }
            colIndices += [-1]
            for c in (self.columns-WIDE..<self.columns) { colIndices += [c] }
        } else {
            for c in (0..<self.columns) { colIndices += [c] }
        }
        
        for row in rowIndices {
            if row == -1 {
                output += "\t• • •\n"
                continue
            }
            
            for col in colIndices {
                if col == -1 {
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
