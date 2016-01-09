//
//  ActivationTests.swift
//  MLKit
//
//  Created by Kesav Mulakaluri on 1/1/16.
//  Copyright © 2016 Kesav Mulakaluri. All rights reserved.
//

import XCTest
@testable import MLKit

class ActivationTests: XCTestCase {
    
    // MARK: - Sigmoid Tests
    
    func testSigmoidCPU() {
        MLSetComputeMode(.CPU)
        let sigmoid = Sigmoid(name: "sig1")
        let m: Matrix = [[0,1], [2,3]]
        let res = sigmoid.apply(m)
        
        for i in 0..<res.elements.count {
            let e = m.elements[i]
            let s = 1/(1+expf(e * -1))
            XCTAssertEqual(s, res.elements[i])
        }
    }
    
    func testSigmoidGPU() {
        MLSetComputeMode(.GPU)
        let sigmoid = Sigmoid(name: "sig1")
        let m: Matrix = [[0,1], [2,3]]
        let res = sigmoid.apply(m)
        
        for i in 0..<res.elements.count {
            let e = m.elements[i]
            let s = 1/(1+expf(e * -1))
            XCTAssertTrue(s - res.elements[i] < 0.0001)
        }
    }
    
    func testSigmoidDerivativeCPU() {
        MLSetComputeMode(.CPU)
        let sigmoid = Sigmoid(name: "sig1")
        let m: Matrix = [[0,1], [2,3]]
        let res = sigmoid.applyDerivative(m)
        
        for i in 0..<res.elements.count {
            let e = m.elements[i]
            let s = 1/(1+expf(e * -1))
            let d = s * (1-s)
            XCTAssertEqual(d, res.elements[i])
        }
    }
    
    func testSigmoidDerivativeGPU() {
        MLSetComputeMode(.GPU)
        let sigmoid = Sigmoid(name: "sig1")
        let m: Matrix = [[0,1], [2,3]]
        let res = sigmoid.applyDerivative(m)
        
        for i in 0..<res.elements.count {
            let e = m.elements[i]
            let s = 1/(1+expf(e * -1))
            let d = s * (1-s)
            XCTAssertTrue(d - res.elements[i] < 0.0001)
        }
    }
    
    // MARK: - Tanh Tests
    
    func testTanhCPU() {
        MLSetComputeMode(.CPU)
        let tanh_b = Tanh(name: "tanh1")
        let m: Matrix = [[0,1], [2,3]]
        let res = tanh_b.apply(m)
        
        for i in 0..<res.elements.count {
            let e = m.elements[i]
            let s = tanh(e)
            XCTAssertEqual(s, res.elements[i])
        }
    }
    
    func testTanhGPU() {
        MLSetComputeMode(.GPU)
        let tanh_b = Tanh(name: "tanh1")
        let m: Matrix = [[0,1], [2,3]]
        let res = tanh_b.apply(m)
        
        for i in 0..<res.elements.count {
            let e = m.elements[i]
            let s = tanh(e)
            XCTAssertTrue(s - res.elements[i] < 0.0001)
        }
    }
    
    func testTanhDerivativeCPU() {
        MLSetComputeMode(.CPU)
        let tanh_b = Tanh(name: "tanh1")
        let m: Matrix = [[0,1], [2,3]]
        let res = tanh_b.applyDerivative(m)
        
        for i in 0..<res.elements.count {
            let e = m.elements[i]
            let s = tanh(e)
            let d = 1 - powf(s, 2.0)
            XCTAssertTrue(d - res.elements[i] < 0.0001)
        }
    }
    
    func testTanhDerivativeGPU() {
        MLSetComputeMode(.GPU)
        let tanh_b = Tanh(name: "tanh1")
        let m: Matrix = [[0,1], [2,3]]
        let res = tanh_b.applyDerivative(m)
        
        for i in 0..<res.elements.count {
            let e = m.elements[i]
            let s = tanh(e)
            let d = 1 - powf(s, 2.0)
            XCTAssertTrue(d - res.elements[i] < 0.0001)
        }
    }
    
    
    // MARK: - ReLU Tests
    
    func testReluCPU() {
        MLSetComputeMode(.CPU)
        let relu = ReLU(name: "relu1")
        let m: Matrix = [[0,-1], [2,3]]
        let res = relu.apply(m)
        
        for i in 0..<res.elements.count {
            let e = m.elements[i]
            if (e > 0) {
                XCTAssertEqual(e, res.elements[i])
            } else {
                XCTAssertEqual(0, res.elements[i])
            }
        }
    }
    
    func testReluGPU() {
        MLSetComputeMode(.GPU)
        let relu = ReLU(name: "relu1")
        let m: Matrix = [[0,-1], [2,3]]
        let res = relu.apply(m)
        
        for i in 0..<res.elements.count {
            let e = m.elements[i]
            if (e > 0) {
                XCTAssertEqual(e, res.elements[i])
            } else {
                XCTAssertEqual(0, res.elements[i])
            }
        }
    }
    
    func testReluDerivativeCPU() {
        MLSetComputeMode(.CPU)
        let relu = ReLU(name: "relu1")
        let m: Matrix = [[0,1], [2,3]]
        let res = relu.applyDerivative(m)
        
        XCTAssertEqual(0, res.elements[0])
        XCTAssertEqual(1, res.elements[1])
        XCTAssertEqual(1, res.elements[2])
        XCTAssertEqual(1, res.elements[3])
    }
    
    func testReluDerivativeGPU() {
        MLSetComputeMode(.GPU)
        let relu = ReLU(name: "relu1")
        let m: Matrix = [[0,1], [2,3]]
        let res = relu.applyDerivative(m)
        
        XCTAssertEqual(0, res.elements[0])
        XCTAssertEqual(1, res.elements[1])
        XCTAssertEqual(1, res.elements[2])
        XCTAssertEqual(1, res.elements[3])
    }
}
