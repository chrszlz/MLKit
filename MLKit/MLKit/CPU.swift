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
        cblas_saxpy(Int32(res.elements.count), 1.0, a.elements, 1, &res.elements, 1)
 
        return res
    }
    
    /// Returns `a` - `b`.
    func subtractMatrices(a a: Matrix, b: Matrix) -> Matrix {
        let negative_b = scaleMatrix(b, by: -1.0)
        let res = addMatrices(a: a, b: negative_b)

        return res
    }
    
    /// Returns `a` * `b`.
    func multiplyMatrices(a a: Matrix, b: Matrix) -> Matrix {
        var res = Matrix.zeros(a.rows, columns: b.columns)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(a.rows), Int32(b.columns), Int32(a.columns), 1.0, a.elements, Int32(a.columns), b.elements, Int32(b.columns), 0.0, &res.elements, Int32(res.columns))

        return res
    }
    
    /// Multiplies each element in `a` by `c`.
    func scaleMatrix(a: Matrix, by c: Float) -> Matrix {
        var res = a
        var alpha = c
        cblas_cscal(Int32(a.elements.count), &alpha, &res, 1)
        
        return res
    }
}