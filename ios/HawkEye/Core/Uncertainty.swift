import Foundation

enum Call: String, Codable { case IN, OUT, ABSTAIN
    var cn: String { ["IN": "界内", "OUT": "出界", "ABSTAIN": "判不了"][rawValue]! }
}

struct LandingResult: Codable {
    var courtXY: Pt
    var cov: [Double]            // 2x2 row-major, m^2
    var signedDistance: Double   // m to the nearest boundary edge, + inside
    var distanceSigma: Double
    var nearestEdge: Edge
    var pIn: Double
    var call: Call
    var abstainReason: String?
    var fractionAboveHorizon: Double
}

/// Monte Carlo landing location (port of prototype/uncertainty.py): homography
/// samples x pixel noise; P(in) = share of samples inside the court.
enum Uncertainty {
    /// Pixel covariance: detection noise plus the uniform one-frame motion term v v^T / 12.
    static func pixelCov(detSigma: Double, velocity: Pt?) -> [Double] {
        var c = [detSigma * detSigma, 0, 0, detSigma * detSigma]
        if let v = velocity {
            c[0] += v.x * v.x / 12; c[1] += v.x * v.y / 12; c[2] += v.x * v.y / 12; c[3] += v.y * v.y / 12
        }
        return c
    }

    static func locate(_ cal: Calibration, pixel: Pt, cov: [Double], mode: Mode, contactRadius: Double = 0,
                       alpha: Double = 0.05, n: Int = 2000, seed: UInt64 = 1) -> LandingResult {
        // Cholesky of the 2x2 pixel covariance.
        let l11 = cov[0].squareRoot(), l21 = cov[2] / max(l11, 1e-12), l22 = max(cov[3] - l21 * l21, 0).squareRoot()
        var rng = Rng(seed: seed)
        var pts: [Pt] = [], above = 0, inside = 0
        pts.reserveCapacity(n)
        for i in 0..<n {
            let z1 = rng.normal(), z2 = rng.normal()
            let px = Pt(pixel.x + l11 * z1, pixel.y + l21 * z1 + l22 * z2)
            if !cal.onGround(px) { above += 1 }
            let g = cal.samples[i % cal.samples.count].project(px)
            pts.append(g)
            if Court.isInside(g, mode, contactRadius: contactRadius) { inside += 1 }
        }
        let mx = pts.map(\.x).reduce(0, +) / Double(n), my = pts.map(\.y).reduce(0, +) / Double(n)
        var cxx = 0.0, cxy = 0.0, cyy = 0.0
        for p in pts { cxx += (p.x - mx) * (p.x - mx); cxy += (p.x - mx) * (p.y - my); cyy += (p.y - my) * (p.y - my) }
        cxx /= Double(n - 1); cxy /= Double(n - 1); cyy /= Double(n - 1)
        let centre = cal.toCourt(pixel)
        let (d, edge) = Court.signedDistance(centre, mode)
        let dS = pts.map { Court.signedDistance($0, mode).d }
        let dm = dS.reduce(0, +) / Double(n)
        let dSigma = (dS.map { ($0 - dm) * ($0 - dm) }.reduce(0, +) / Double(n - 1)).squareRoot()
        let pIn = Double(inside) / Double(n), fAbove = Double(above) / Double(n)
        var call: Call = pIn >= 1 - alpha ? .IN : (pIn <= alpha ? .OUT : .ABSTAIN)
        var reason: String? = call == .ABSTAIN ? String(format: "P(界内)=%.2f，离线太近", pIn) : nil
        if fAbove > 0.01 {
            call = .ABSTAIN
            reason = String(format: "%.0f%% 的样本越过了画面里地面的地平线，离相机太远", fAbove * 100)
        }
        return LandingResult(courtXY: centre, cov: [cxx, cxy, cxy, cyy], signedDistance: d + contactRadius, distanceSigma: dSigma,
                             nearestEdge: edge, pIn: pIn, call: call, abstainReason: reason, fractionAboveHorizon: fAbove)
    }

    /// 95% ellipse in court metres (for drawing).
    static func ellipse(centre: Pt, cov: [Double], k: Double = 2.4477, steps: Int = 48) -> [Pt] {
        let a = cov[0], b = cov[1], d = cov[3]
        let tr = a + d, det = a * d - b * b
        let l1 = tr / 2 + (tr * tr / 4 - det).squareRoot(), l2 = max(tr / 2 - (tr * tr / 4 - det).squareRoot(), 0)
        let ang = 0.5 * atan2(2 * b, a - d)
        return (0...steps).map { i in
            let t = 2 * Double.pi * Double(i) / Double(steps)
            let ex = k * l1.squareRoot() * cos(t), ey = k * l2.squareRoot() * sin(t)
            return Pt(centre.x + ex * cos(ang) - ey * sin(ang), centre.y + ex * sin(ang) + ey * cos(ang))
        }
    }
}
