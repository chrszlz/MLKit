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
        MLSetComputeMode(.cpu)
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
        MLSetComputeMode(.gpu)
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
        MLSetComputeMode(.cpu)
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
        MLSetComputeMode(.gpu)
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
        MLSetComputeMode(.cpu)
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
        MLSetComputeMode(.gpu)
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
        MLSetComputeMode(.cpu)
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
        MLSetComputeMode(.gpu)
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
        MLSetComputeMode(.cpu)
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
        MLSetComputeMode(.gpu)
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
        MLSetComputeMode(.cpu)
        let relu = ReLU(name: "relu1")
        let m: Matrix = [[0,1], [2,3]]
        let res = relu.applyDerivative(m)
        
        XCTAssertEqual(0, res.elements[0])
        XCTAssertEqual(1, res.elements[1])
        XCTAssertEqual(1, res.elements[2])
        XCTAssertEqual(1, res.elements[3])
    }
    
    func testReluDerivativeGPU() {
        MLSetComputeMode(.gpu)
        let relu = ReLU(name: "relu1")
        let m: Matrix = [[0,1], [2,3]]
        let res = relu.applyDerivative(m)
        
        XCTAssertEqual(0, res.elements[0])
        XCTAssertEqual(1, res.elements[1])
        XCTAssertEqual(1, res.elements[2])
        XCTAssertEqual(1, res.elements[3])
    }
    
    
    // MARK: - Softmax Tests
    
    func testSoftmaxCPU() {
        MLSetComputeMode(.cpu)
        let softmax = Softmax(name: "softmax")
        let m: Matrix = [[0,1], [2,3]]
        let m_exp = m.exp()
        let sum = m_exp.sum()
        let res = softmax.apply(m)
        
        for i in 0..<m.elements.count {
            XCTAssertTrue(res.elements[i] - (m_exp.elements[i] * 1/sum) < 0.0001)
        }
    }
    
    func testSoftmaxGPU() {
        MLSetComputeMode(.gpu)
        let softmax = Softmax(name: "softmax")
        let m: Matrix = [[0,1], [2,3]]
        let m_exp = m.exp()
        let sum = m_exp.sum()
        let res = softmax.apply(m)
        
        for i in 0..<m.elements.count {
            XCTAssertTrue(res.elements[i] - (m_exp.elements[i] * 1/sum) < 0.0001)
        }
    }
}
