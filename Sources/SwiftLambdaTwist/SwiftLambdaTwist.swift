//
//  LambdaTwist.swift
//
//  Created by Tuukka Karvonen on 2020/12/03.
//

import Foundation
import simd

extension simd_float3 {
    func norm() -> simd_float1 {
        return sqrt(self.squaredNorm())
    }
    
    func squaredNorm() -> simd_float1 {
        return (self.x * self.x) + (self.y * self.y) + (self.z * self.z)
    }
    
    func dot(other: simd_float3) -> simd_float1 {
        return self.x * other.x + self.y * other.y + self.z * other.z
    }
    
    func cross(other: simd_float3) -> simd_float3 {
        return simd_float3(self.y * other.z - self.z * other.y, self.z * other.x - self.x * other.z, self.x * other.y - self.y * other.x)
    }
}

// lambdatwist algorithm
// ref: https://github.com/midjji/lambdatwist-p3p
// ref: https://github.com/vlarsson/lambdatwist
// ref: https://openaccess.thecvf.com/content_ECCV_2018/papers/Mikael_Persson_Lambda_Twist_An_ECCV_2018_paper.pdf
public func lambdaTwist(Xs: [simd_float3], Ys: [simd_float3]) -> [(simd_float3x3, simd_float3)] {
    var ret: [(simd_float3x3, simd_float3)] = []
    
    // 3: Normalize yi = yi/|yi|
    let normalizedYs: [simd_float3] = Ys.map({
        $0 / $0.norm()
    })
    
    // 4: Compute aij and bij according to (3)
    // (3): |xi−xj| = aij
    //      bij =yiyj.dot()
    let a12 = (Xs[0] - Xs[1]).squaredNorm()
    let a13 = (Xs[0] - Xs[2]).squaredNorm()
    let a23 = (Xs[1] - Xs[2]).squaredNorm()
    
    let b12 = normalizedYs[0].dot(other: normalizedYs[1])
    let b13 = normalizedYs[0].dot(other: normalizedYs[2])
    let b23 = normalizedYs[1].dot(other: normalizedYs[2])

    // 5: Construct D1 and D2 from (5) and (6)
    // (5): D1 = M12a23 − M23a12 = 􏰀d11d12d13􏰁
    // (6): D2 = M13a23 − M23a13 = 􏰀d21d22d23􏰁
    
    // create M12, M13 and M23
    let M12: simd_float3x3 = simd_float3x3(simd_float3(1, -b12, 0), simd_float3(-b12, 1, 0), simd_float3(0, 0, 0))
    let M13: simd_float3x3 = simd_float3x3(simd_float3(1, 0, -b13), simd_float3(0, 0, 0), simd_float3(-b13, 0, 1))
    let M23: simd_float3x3 = simd_float3x3(simd_float3(0, 0, 0), simd_float3(0, 1, -b23), simd_float3(0, -b23, 1))
    
    let D1 = M12 * a23 - M23 * a12
    let D2 = M13 * a23 - M23 * a13
    
    // 6: Compute a real root γ to (8)-(10) of the cubic equation
    
    // get coefficients c3, c2, c1, c0
    let c3 = D2.determinant
    let c0 = D1.determinant
    
    // NOTE: c1 and c2 are switched in the paper
    // c1 = dT21(d12 ×d13) + dT22(d13 ×d11) + dT23(d11 ×d12)
    let c1 =
        D2.columns.0.dot(other: D1.columns.1.cross(other: D1.columns.2)) +
        D2.columns.1.dot(other: D1.columns.2.cross(other: D1.columns.0)) +
        D2.columns.2.dot(other: D1.columns.0.cross(other: D1.columns.1))
    
    // c2 = dT11(d22 ×d23) + dT12(d23 ×d21) + dT13(d21 ×d22)
    let c2 =
        D1.columns.0.dot(other: D2.columns.1.cross(other: D2.columns.2)) +
        D1.columns.1.dot(other: D2.columns.2.cross(other: D2.columns.0)) +
        D1.columns.2.dot(other: D2.columns.0.cross(other: D2.columns.1))
    
    let A = Double(c2 / c3)
    let B = Double(c1 / c3)
    let C = Double(c0 / c3)
    
    let cubicRoot = solveOneCubic(a: A, b: B, c: C)
    
    // 7: D0 = D1 + γD2
    let D0 = D1 + simd_float1(cubicRoot) * D2
    
    // 8: [E,σ1,σ2]=EIG3X3KNOWN0(D0).SeeAlgorithm 2
    let Es1s2 = eig3x3Known0(D0: D0)
    
    // 9: s = ±sqrt(􏰄−σ2/σ1)
    let spos = sqrt(max(0, -Es1s2.2/Es1s2.1))
    let sneg = -spos
    
    // 10: Compute the τk > 0, τk ∈ R for each s using Eqn (14) with coefficients in Eqn (15)
    // spos
    let tauspos = calcTaus(s: spos, E: Es1s2.0, a12: a12, a13: a13, a23: a23, b12: b12, b13: b13, b23: b23)
    let tausneg = calcTaus(s: sneg, E: Es1s2.0, a12: a12, a13: a13, a23: a23, b12: b12, b13: b13, b23: b23)

    // 11: Compute Λk according to Eqn (16), λ3k = τkλ2k and Eqn (13), λ1k > 0
    // so for each τk we get Λk
    let E = Es1s2.0
    
    var Λs: [simd_float3] = []
    for τ in tauspos {
        // NOTE: here also differs from paper
        let Λ2k = sqrt(a23 / (τ * (τ - 2.0 * b23) + 1))
        let Λ3k = τ * Λ2k
        // Λ1k
        let w0 = (E[1,0] - spos * E[1,1]) / (spos * E[0,1] - E[0,0])
        let w1 = (E[2,0] - spos * E[2,1]) / (spos * E[0,1] - E[0,0])
        let Λ1k = w0 * Λ2k + w1 * Λ3k
        guard Λ1k >= 0 else { continue }
        Λs.append(simd_float3(Λ1k, Λ2k, Λ3k))
    }
    for τ in tausneg {
        let Λ2k = sqrt(a23 / (τ * (τ - 2.0 * b23) + 1))
        let Λ3k = τ * Λ2k
        // Λ1k
        let w0 = (E[1,0] - sneg * E[1,1]) / (sneg * E[0,1] - E[0,0])
        let w1 = (E[2,0] - sneg * E[2,1]) / (sneg * E[0,1] - E[0,0])
        let Λ1k = w0 * Λ2k + w1 * Λ3k
        guard Λ1k >= 0 else { continue }
        Λs.append(simd_float3(Λ1k, Λ2k, Λ3k))
    }
    
    let X = simd_float3x3(Xs[0]-Xs[1], Xs[0]-Xs[2], (Xs[0]-Xs[1]).cross(other: (Xs[0]-Xs[2])))
    
    let Xinv = X.inverse
    
    // 13: for each valid Λk do
    for Λ in Λs {
        // 14: Gauss-Newton-Refine(Λk), see Section 3.8
        let Λref = gaussNewtonRefine(Λ: Λ, a12: a12, a13: a13, a23: a23, b12: b12, b13: b13, b23: b23)
        // 15: Yk = MIX(λ1ky1 − λ2ky2, λ1ky1 − λ3ky3)
        let Y0 = Λref[0] * normalizedYs[0] - Λref[1] * normalizedYs[1]
        let Y1 = Λref[0] * normalizedYs[0] - Λref[2] * normalizedYs[2]
        let Y2 = Y0.cross(other: Y1)
        
        // 16: Rk =YkXinv
        let R = simd_float3x3(Y0, Y1, Y2) * Xinv
        
        // 17: tk =λ1ky1 −Rkx1
        let t = Λref[0] * normalizedYs[0] - R * Xs[0]
        ret.append((R, t))
    }
    
    return ret
}

func gaussNewtonRefine(Λ: simd_float3,
                       a12: simd_float1,
                       a13: simd_float1,
                       a23: simd_float1,
                       b12: simd_float1,
                       b13: simd_float1,
                       b23: simd_float1) -> simd_float3 {
    let iterations = 2
    
    var Λiter = Λ
    for i in 0..<iterations {
        let Λ0 = Λ[0]
        let Λ1 = Λ[1]
        let Λ2 = Λ[2]
        
        // write the equations out
        let res0 = Λ0 * Λ0 + Λ1 * Λ1 - 2 * Λ0 * Λ1 * b12 - a12
        let res1 = Λ0 * Λ0 + Λ2 * Λ2 - 2 * Λ0 * Λ2 * b13 - a13
        let res2 = Λ1 * Λ1 + Λ2 * Λ2 - 2 * Λ1 * Λ2 * b23 - a23
        
        // calculate the jacobian
        let dres0dΛ0 = 2 * Λ0 - 2 * Λ1 * b12
        let dres0dΛ1 = 2 * Λ1 - 2 * Λ0 * b12
        let dres1dΛ0 = 2 * Λ0 - 2 * Λ2 * b13
        let dres1dΛ2 = 2 * Λ2 - 2 * Λ0 * b13
        let dres2dΛ1 = 2 * Λ1 - 2 * Λ2 * b23
        let dres2dΛ2 = 2 * Λ2 - 2 * Λ1 * b23
        
        // now we ca ncalculate Λ's next iteration by:
        // Λn+1 = Λn - J^-1 * res
        let J = simd_float3x3(simd_float3(dres0dΛ0, dres0dΛ1, 0), simd_float3(dres1dΛ0, 0, dres1dΛ2), simd_float3(0, dres2dΛ1, dres2dΛ2))
        
        Λiter = Λiter - J.inverse * simd_float3(res0, res1, res2)
    }
    return Λiter
}

func calcTaus(s: simd_float1, E: simd_float3x3, a12: simd_float1, a13: simd_float1, a23: simd_float1, b12: simd_float1, b13: simd_float1, b23: simd_float1) -> [simd_float1] {
    // calc ws
    let w0 = (E[1,0] - s * E[1,1]) / (s * E[0,1] - E[0,0])
    let w1 = (E[2,0] - s * E[2,1]) / (s * E[0,1] - E[0,0])
    
    // calc coeffiecients
    // NOTE: paper/code is different, investigate why
    let a = ((a13 - a12)*(w1*w1) + 2*a12*b13*w1 - a12)
    let b = (2*a12*b13*w0 - 2*a13*b12*w1 - 2*w0*w1*(a12-a13))
    let c = ((a13-a12)*w0*w0 - 2*a13*b12*w0)+a13
    
    // solve quadratic
    let taus = solveQuadratic(a: Double(a), b: Double(b), c: Double(c))
    
    let flotaus = taus.map({
        simd_float1($0)
    })
    
    return flotaus
}

func eig3x3Known0(D0: simd_float3x3) -> (simd_float3x3, simd_float1, simd_float1) {
    var b3 = D0.columns.1.cross(other: D0.columns.2)
    b3 = b3 / b3.norm()
    // NOTE: this was wrong in original paper (missing minus in front of D0[0,0])
    let p1 = -D0[0,0] - D0[1,1] - D0[2,2]
    // NOTE: this was wrong in original paper (presented as m_1 * m_5 + m_9 when in fact should be m_1 * (m_5 + m_9)
    let p0 = -(D0[0,1] * D0[0,1]) - (D0[0,2] * D0[0,2]) - (D0[1,2] * D0[1,2]) + D0[0,0] * (D0[1,1] + D0[2,2]) + D0[1,1] * D0[2,2]
    let s1s2 = solveQuadratic(a: 1.0, b: Double(p1), c: Double(p0))
    let b1 = getEigVector(m: D0, r: simd_float1(s1s2[0]))
    let b2 = getEigVector(m: D0, r: simd_float1(s1s2[1]))
    
    if abs(s1s2[0]) > abs(s1s2[1]) {
        // NOTE: notation had row-major/column-major switch here?
        return (simd_float3x3(b1, b2, b3).transpose, simd_float1(s1s2[0]), simd_float1(s1s2[1]))
    }
    return (simd_float3x3(b2, b1, b3).transpose, simd_float1(s1s2[1]), simd_float1(s1s2[0]))
}

func getEigVector(m: simd_float3x3, r: simd_float1) -> simd_float3 {
    let r2 = r * r
    let m22 = m[0,1] * m[0,1]
    let m1m5 = m[0,0] * m[1,1]
    let cinner = r * (m[0,0] + m[1,1])
    let c = (r2 + m1m5 - cinner - m22)
    let a1 = (r * m[0,2] + m[0,1] * m[1,2] - m[0,2]*m[1,1]) / c
    let a2 = (r * m[1,2] + m[0,1] * m[0,2] - m[0,0] * m[1,2]) / c
    var v = simd_float3(a1, a2, 1.0)
    v = v / v.norm()
    return v
}

// ref: https://www.e-education.psu.edu/png520/m11_p6.html
func solveOneCubic(a: Double, b: Double, c: Double) -> Double {
    let Q = (a*a - 3*b) / 9.0
    let R = (2 * a * a * a - 9 * a * b + 27 * c) / 54.0
    let QQQ = Q * Q * Q
    let M = R * R - QQQ
    if M < 0 {
        let theta = acos(R / sqrt(Q*Q*Q))
        return -(2*sqrt(Q)*cos(theta/3)) - a / 3.0
    }
    let _s = -R + sqrt(M)
    let _t = -R - sqrt(M)
    let S = _s < 0 ? -pow(-_s, 1.0/3.0) : pow(_s, 1.0/3.0)
    let T = _t < 0 ? -pow(-_t, 1.0/3.0) : pow(_s, 1.0/3.0)
    //S = _s < 0 ? -pow(-_s, 1.0/3.0) : pow(_s, 1.0/3.0)
    //T = _t < 0 ? -pow(-_t, 1.0/3.0) : pow(_s, 1.0/3.0)
    //let S = pow(-R + sqrt(M), 1.0/3.0)
    //let T = pow(-R - sqrt(M), 1.0/3.0)
    return S + T - a / 3.0
}

func solveQuadratic(a: Double, b: Double, c: Double) -> [Double] {
    let D = b*b - 4*a*c
    
    guard D >= 0 else { return [] }
        
    guard D > 0 else { return [-b / 2*a] }
    let sD = sqrt(D)
    return [(-b + sD) / (2*a), (-b - sD) / (2*a)]
}
