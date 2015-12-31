//
//  Shape.swift
//  MLKit
//
//  Created by Chris Zelazo on 12/22/15.
//  Copyright © 2015 Kesav Mulakaluri. All rights reserved.
//

import Foundation

public struct Shape : Equatable {
    /// The number of rows in this matrix.
    public let rows: Int
    
    /// The number of columns in this matrix.
    public let columns: Int
    
    /// The designated initializer. Creates a Shape with specified `rows` and `columns`.
    public init(_ rows: Int, _ columns: Int) {
        self.rows = rows
        self.columns = columns
    }
}

// MARK: - Equatable
public func == (lhs: Shape, rhs: Shape) -> Bool {
    return (lhs.rows == rhs.rows && lhs.columns == rhs.columns)
}

// MARK - Printable
extension Shape: CustomStringConvertible {
    public var description: String {
        return "(\(self.rows), \(self.columns))"
    }
}