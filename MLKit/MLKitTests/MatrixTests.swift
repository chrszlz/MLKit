//
//  MatrixTests.swift
//  MLKit
//
//  Created by Kesav Mulakaluri on 12/23/15.
//  Copyright © 2015 Kesav Mulakaluri. All rights reserved.
//

import XCTest
@testable import MLKit

class MatrixTests: XCTestCase {
    
    
    // MARK: - Matrix Addition Tests
    
    func testMatrixAddCPU() {
        MLSetComputeMode(.cpu)
        let x = Matrix([[1,2], [3,4]])
        let y = Matrix([[1,2], [3,4]])
        let res = x + y
        XCTAssert(res == Matrix([[2,4],[6,8]]))
    }
    
    func testMatrixAddGPU() {
        MLSetComputeMode(.gpu)
        let x = Matrix([[1,2], [3,4]])
        let y = Matrix([[1,2], [3,4]])
        let res = x + y
        XCTAssert(res == Matrix([[2,4],[6,8]]))
    }
    
    
    // MARK: - Matrix Subtraction Tests
    
    func testMatrixSubtractCPU() {
        MLSetComputeMode(.cpu)
        let x = Matrix([[1,2], [3,4]])
        let y = Matrix([[0,1], [2,3]])
        let res = x - y
        XCTAssert(res == Matrix([[1,1],[1,1]]))
    }
    
    func testMatrixSubtractGPU() {
        MLSetComputeMode(.gpu)
        let x = Matrix([[1,2], [3,4]])
        let y = Matrix([[0,1], [2,3]])
        let res = x - y
        XCTAssert(res == Matrix([[1,1],[1,1]]))
    }

    
    // MARK: - Matrix Scale Tests
    
    func testMatrixScaleCPU() {
        MLSetComputeMode(.cpu)
        let x = Matrix.ones(rows: 2, columns: 2)
        let c: Float = 2.0
        let res = c * x
        XCTAssert(res == Matrix([[2,2], [2,2]]))
    }

    func testMatrixScaleGPU() {
        MLSetComputeMode(.gpu)
        let x = Matrix.ones(rows: 2, columns: 2)
        let c: Float = 2.0
        let res = c * x
        XCTAssert(res == Matrix([[2,2], [2,2]]))
    }
    
    
    // MARK: - Matrix Multiplication Tests
    
    func testMatrixMultiplyCPU() {
        MLSetComputeMode(.cpu)
        let x = Matrix([[1,2], [3,4]])
        let y = Matrix([[4,5], [6,7]])
        let res = x * y
        XCTAssert(res == Matrix([[16, 19], [36, 43]]))
    }
    
    
    // MARK: - Matrix Sum Tests
    
    func testMatrixSumCPU() {
        MLSetComputeMode(.cpu)
        let x = Matrix([[1,2], [3,4]])
        let res = x.sum()
        XCTAssertEqual(res, Float(10.0))
    }
    
    // MARK: - Matrix Exp Tests
    
    func testMatrixExpCPU() {
        MLSetComputeMode(.cpu)
        let x = Matrix([[1,2], [3,4]])
        let res = x.exp()
        
        for i in 0..<x.elements.count {
            XCTAssertTrue(res.elements[i] - exp(x.elements[i]) < 0.0001)
        }
    }
    
    func testMatrixExpGPU() {
        MLSetComputeMode(.gpu)
        let x = Matrix([[1,2], [3,4]])
        let res = x.exp()
        
        for i in 0..<x.elements.count {
            XCTAssertTrue(res.elements[i] - exp(x.elements[i]) < 0.0001)
        }
    }
}
