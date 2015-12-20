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

public struct Matrix {
    
    /// The number of rows in this matrix.
    public let rows: Int
    
    /// The number of columns in this matrix.
    public let columns: Int
    
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
    
    /// Initializes the matrix from a 2D array.
    ///
    /// - parameter elements: The 2D array used to initialize the matrix.
    public init(elements: [[Float]]) {
        let rows = elements.count
        let columns = elements[0].count
        let elements = elements.flatMap { $0 }
        
        self.init(rows: rows, columns: columns, elements: elements)
    }
    
    /// Returns a `rows` by `columns` matrix of zeros.
    /// - parameter rows: The number of rows in the matrix.
    /// - parameter: columns: The number of columns in the matrix.
    public static func zeros(rows: Int, columns: Int) -> Matrix {
        let elements = [Float](count: rows * columns, repeatedValue: 0)
        return Matrix(rows: rows, columns: columns, elements: elements)
    }
    
    /// Returns a `rows` by `columns` matrix of ones.
    /// - parameter rows: The number of rows in the matrix.
    /// - parameter columns: The number of columns in the matrix.
    public static func ones(rows: Int, columns: Int) -> Matrix {
        let elements = [Float](count: rows * columns, repeatedValue: 1)
        return Matrix(rows: rows, columns: columns, elements: elements)
    }
    
    /// The designated initializer. Creates a `rows` by `columns` matrix with
    /// `elements`.
    init(rows: Int, columns: Int, elements: [Float]) {
        guard rows > 0 && columns > 0 else {
            fatalError("Invalid dimensions. Matrix must be at least 1x1")
        }
        
        self.rows = rows
        self.columns = columns
        self.elements = elements
    }
    
    
    // MARK: - Operations
    
    /// Returns the transpose of this matrix.
    private func transpose() -> Matrix {
        var res = Matrix.zeros(self.columns, columns: self.rows)
        vDSP_mtrans(self.elements, 1, &(res.elements), 1, vDSP_Length(res.rows), vDSP_Length(res.columns))
        return res
    }
    
    /// Returns the sum of the two matrices (a + b).
    private static func add(a: Matrix, b: Matrix) -> Matrix {
        return a
    }
    
    /// Returns the difference of the two matrices (a - b).
    private static func subtract(a: Matrix, b: Matrix) -> Matrix {
        return a
    }
    
    /// Returns the product of two matrices (a * b).
    private static func multiply(a: Matrix, b: Matrix) -> Matrix {
        return a
    }
    
    /// Returns a `Matrix` where each element in `a` is multiplied by `s`.
    private static func scale(s: Float, a: Matrix) -> Matrix {
        return a
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
