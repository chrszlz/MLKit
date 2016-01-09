//
//  CPU.swift
//  MLKit
//
//  Created by Kesav Mulakaluri on 12/22/15.
//  Copyright © 2015 Kesav Mulakaluri. All rights reserved.
//

import Foundation
import Accelerate

/// CPU is a light weight wrapper around the Accelerate framework to perform
/// fast matrix computations.
class CPU: MLComputeDevice {
    
    /// Returns `a` + `b`.
    func addMatrices(a a: Matrix, b: Matrix) -> Matrix {
        var res = b        
        vDSP_vadd(a.elements, 1, b.elements, 1, &res.elements, 1, vDSP_Length(res.elements.count))
 
        return res
    }
    
    /// Returns `a` - `b`.
    func subtractMatrices(a a: Matrix, b: Matrix) -> Matrix {
        let negative_b = scaleMatrix(b, by: -1.0)
        let res = addMatrices(a: a, b: negative_b)

        return res
    }
    
    /// Multiplies each element in `a` by `c`.
    func scaleMatrix(var a: Matrix, var by c: Float) -> Matrix {
        vDSP_vsmul(a.elements, 1, &c, &a.elements, 1, vDSP_Length(a.elements.count))
        
        return a
    }
    
    
    /// Returns `a` * `b`.
    func multiplyMatrices(a a: Matrix, b: Matrix) -> Matrix {
        var res = Matrix.zeros(rows: a.rows, columns: b.columns)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(a.rows), Int32(b.columns), Int32(a.columns), 1.0, a.elements, Int32(a.columns), b.elements, Int32(b.columns), 0.0, &res.elements, Int32(res.columns))
        
        return res
    }
    
    /// Applies the sigmoid function to each element in `a`.
    func applySigmoid(a: Matrix) -> Matrix {
        var output = Matrix.zeros(a.shape)
        let count = Int32(output.elements.count)
        var one: Float = 1.0
        var negOne: Float = -1.0
        vDSP_vneg(a.elements, 1, &output.elements, 1, vDSP_Length(count))
        vvexpf(&output.elements, output.elements, [count])
        vDSP_vsadd(&output.elements, 1, &one, &output.elements, 1, vDSP_Length(count))
        vvpowsf(&output.elements, &negOne, output.elements, [count])
        return output
    }
    
    /// Applies the derivative of the sigmoid function to each element in `a`.
    func applySigmoidDerivative(a: Matrix) -> Matrix {
        let sigmoid = applySigmoid(a)
        var oneMinusSigmoid = sigmoid
        var one: Float = 1.0
        let count = sigmoid.elements.count
        vDSP_vneg(sigmoid.elements, 1, &oneMinusSigmoid.elements, 1, vDSP_Length(count))
        vDSP_vsadd(oneMinusSigmoid.elements, 1, &one, &oneMinusSigmoid.elements, 1, vDSP_Length(count))
        
        var output = Matrix.zeros(a.shape)
        vDSP_vmul(sigmoid.elements, 1, oneMinusSigmoid.elements, 1, &output.elements, 1, vDSP_Length(count))

        return output
    }
    
    /// Applies the hyperbolic tangent to each element in `a`.
    func applyTanh(a: Matrix) -> Matrix {
        var res = a
        vvtanhf(&res.elements, a.elements, [Int32(a.elements.count)])
        
        return res
    }
    
    /// Applies the derivative of the hyperbolic tangent to each element in `a`.
    func applyTanhDerivative(a: Matrix) -> Matrix {
        var output = applyTanh(a)
        let count = output.elements.count
        var one: Float = 1.0
        var twos = [Float](count: count, repeatedValue: 2)
        vvpowf(&output.elements, &twos, output.elements, [Int32(count)])
        vDSP_vneg(output.elements, 1, &output.elements, 1, vDSP_Length(count))
        vDSP_vsadd(output.elements, 1, &one, &output.elements, 1, vDSP_Length(count))
        
        return output
    }
    
    /// Applies the rectified linear activation function to each element in `a`.
    func applyRelu(a: Matrix) -> Matrix {
        var res = a
        var zero: Float = 0.0
        vDSP_vthres(a.elements, 1, &zero, &res.elements, 1, vDSP_Length(res.elements.count))
        
        return res
    }
    
    /// Applies the Rectified Linear activation derivative to each element in `a`.
    func applyReluDerivative(a: Matrix) -> Matrix {
        var output = a
        
        for i in 0..<a.elements.count {
            output.elements[i] = (a.elements[i] == 0) ? 0 : 1
        }
        
        return output
    }
}