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
}
