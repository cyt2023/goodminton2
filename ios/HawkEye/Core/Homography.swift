import Foundation

/// 3x3 matrix, row-major.
struct Mat3: Codable {
    var m: [Double]
    init(_ m: [Double]) { precondition(m.count == 9); self.m = m }
    static let identity = Mat3([1, 0, 0, 0, 1, 0, 0, 0, 1])

    func apply(_ p: Pt) -> (Pt, w: Double) {
        let X = m[0] * p.x + m[1] * p.y + m[2], Y = m[3] * p.x + m[4] * p.y + m[5], w = m[6] * p.x + m[7] * p.y + m[8]
        return (Pt(X / w, Y / w), w)
    }
    func project(_ p: Pt) -> Pt { apply(p).0 }

    static func * (a: Mat3, b: Mat3) -> Mat3 {
        var r = [Double](repeating: 0, count: 9)
        for i in 0..<3 { for j in 0..<3 { for k in 0..<3 { r[i * 3 + j] += a.m[i * 3 + k] * b.m[k * 3 + j] } } }
        return Mat3(r)
    }

    var inverse: Mat3? {
        let a = m
        let c00 = a[4] * a[8] - a[5] * a[7], c01 = a[5] * a[6] - a[3] * a[8], c02 = a[3] * a[7] - a[4] * a[6]
        let det = a[0] * c00 + a[1] * c01 + a[2] * c02
        guard abs(det) > 1e-15, det.isFinite else { return nil }
        let inv = [c00, a[2] * a[7] - a[1] * a[8], a[1] * a[5] - a[2] * a[4],
                   c01, a[0] * a[8] - a[2] * a[6], a[2] * a[3] - a[0] * a[5],
                   c02, a[1] * a[6] - a[0] * a[7], a[0] * a[4] - a[1] * a[3]].map { $0 / det }
        return Mat3(inv)
    }
    var normalized: Mat3 { Mat3(m.map { $0 / m[8] }) }
}

enum Homography {
    /// Least-squares homography mapping src -> dst (>= 4 points), with Hartley
    /// normalisation and h33 = 1. Returns nil for degenerate input.
    static func fit(src: [Pt], dst: [Pt], refine doRefine: Bool = true) -> Mat3? {
        guard src.count == dst.count, src.count >= 4 else { return nil }
        guard let (Ts, s) = normalise(src), let (Td, d) = normalise(dst) else { return nil }
        // Normal equations A^T A h = A^T b for the 8 unknowns.
        var ata = [Double](repeating: 0, count: 64), atb = [Double](repeating: 0, count: 8)
        for (p, q) in zip(s, d) {
            let rows: [([Double], Double)] = [
                ([p.x, p.y, 1, 0, 0, 0, -p.x * q.x, -p.y * q.x], q.x),
                ([0, 0, 0, p.x, p.y, 1, -p.x * q.y, -p.y * q.y], q.y)]
            for (r, b) in rows {
                for i in 0..<8 {
                    atb[i] += r[i] * b
                    for j in 0..<8 { ata[i * 8 + j] += r[i] * r[j] }
                }
            }
        }
        guard let h = solve(ata, atb, 8) else { return nil }
        let Hn = Mat3(h + [1])
        guard let TdInv = Td.inverse else { return nil }
        let H = (TdInv * Hn * Ts).normalized
        guard H.m.allSatisfy({ $0.isFinite }) else { return nil }
        return doRefine && src.count > 4 ? refine(H, src: src, dst: dst) : H
    }

    /// Levenberg-Marquardt on the reprojection error in dst coordinates (as OpenCV does
    /// after the linear estimate), 8 parameters with h33 = 1.
    static func refine(_ H0: Mat3, src: [Pt], dst: [Pt], iterations: Int = 20) -> Mat3 {
        func residuals(_ h: [Double]) -> [Double] {
            let H = Mat3(h + [1])
            return zip(src, dst).flatMap { p, q -> [Double] in let r = H.project(p); return [r.x - q.x, r.y - q.y] }
        }
        func cost(_ r: [Double]) -> Double { r.reduce(0) { $0 + $1 * $1 } }
        var h = Array(H0.m[0..<8]), r = residuals(h), c = cost(r), lambda = 1e-3
        for _ in 0..<iterations {
            // Numeric Jacobian (2n x 8).
            var J = [[Double]](repeating: [Double](repeating: 0, count: 8), count: r.count)
            for j in 0..<8 {
                let eps = max(abs(h[j]) * 1e-6, 1e-9)
                var hp = h; hp[j] += eps
                let rp = residuals(hp)
                for i in 0..<r.count { J[i][j] = (rp[i] - r[i]) / eps }
            }
            var jtj = [Double](repeating: 0, count: 64), jtr = [Double](repeating: 0, count: 8)
            for i in 0..<r.count { for a in 0..<8 { jtr[a] -= J[i][a] * r[i]; for b in 0..<8 { jtj[a * 8 + b] += J[i][a] * J[i][b] } } }
            var improved = false
            for _ in 0..<8 {
                var A = jtj
                for a in 0..<8 { A[a * 8 + a] *= 1 + lambda }
                guard let step = solve(A, jtr, 8) else { lambda *= 10; continue }
                let hn = zip(h, step).map { $0 + $1 }
                let rn = residuals(hn), cn = cost(rn)
                if cn.isFinite && cn < c { h = hn; r = rn; lambda = max(lambda / 10, 1e-12); improved = c - cn > 1e-12 * c; c = cn; break }
                lambda *= 10
            }
            if !improved { break }
        }
        return Mat3(h + [1])
    }

    private static func normalise(_ pts: [Pt]) -> (Mat3, [Pt])? {
        let n = Double(pts.count)
        let cx = pts.map(\.x).reduce(0, +) / n, cy = pts.map(\.y).reduce(0, +) / n
        let meanDist = pts.map { (($0.x - cx) * ($0.x - cx) + ($0.y - cy) * ($0.y - cy)).squareRoot() }.reduce(0, +) / n
        guard meanDist > 1e-12 else { return nil }
        let k = 2.0.squareRoot() / meanDist
        let T = Mat3([k, 0, -k * cx, 0, k, -k * cy, 0, 0, 1])
        return (T, pts.map { T.project($0) })
    }

    /// Gaussian elimination with partial pivoting.
    static func solve(_ A: [Double], _ b: [Double], _ n: Int) -> [Double]? {
        var a = A, x = b
        for c in 0..<n {
            var piv = c
            for r in (c + 1)..<n where abs(a[r * n + c]) > abs(a[piv * n + c]) { piv = r }
            guard abs(a[piv * n + c]) > 1e-12 else { return nil }
            if piv != c {
                for j in 0..<n { a.swapAt(c * n + j, piv * n + j) }
                x.swapAt(c, piv)
            }
            for r in (c + 1)..<n {
                let f = a[r * n + c] / a[c * n + c]
                if f == 0 { continue }
                for j in c..<n { a[r * n + j] -= f * a[c * n + j] }
                x[r] -= f * x[c]
            }
        }
        for c in stride(from: n - 1, through: 0, by: -1) {
            var s = x[c]
            for j in (c + 1)..<n { s -= a[c * n + j] * x[j] }
            x[c] = s / a[c * n + c]
        }
        return x
    }
}

/// Seeded normal random numbers (reproducible Monte Carlo).
struct Rng {
    var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E3779B97F4A7C15 }
    mutating func next() -> UInt64 {
        state = state &+ 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func uniform() -> Double { Double(next() >> 11) / Double(1 << 53) }
    mutating func normal() -> Double {
        let u1 = max(uniform(), 1e-300), u2 = uniform()
        return (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
    }
}

/// Ground-plane calibration from clicked court keypoints, with Monte Carlo samples.
struct Calibration {
    let H: Mat3          // image px -> court m
    let Hinv: Mat3       // court m -> image px
    let samples: [Mat3]  // image -> court, refitted after perturbing the clicks
    let wSign: Double, wcSign: Double
    let residualRMS: Double, clickSigma: Double, nPoints: Int

    init?(points: [String: Pt], clickSigmaPrior: Double = 1.5, nSamples: Int = 200, seed: UInt64 = 0) {
        let names = points.keys.filter { Court.keypointByName[$0] != nil }.sorted()
        guard names.count >= 4 else { return nil }
        let img = names.map { points[$0]! }, world = names.map { Court.keypointByName[$0]! }
        // Fit court -> image so the error is in pixels, where the click noise is.
        guard let Hci = Homography.fit(src: world, dst: img), let Hic = Hci.inverse?.normalized else { return nil }
        let resid = zip(world, img).map { (Hci.project($0.0) - $0.1).norm }
        let dof = 2 * names.count - 8
        let est = dof > 0 ? (resid.map { $0 * $0 }.reduce(0, +) / Double(dof)).squareRoot() : 0
        let sigma = max(clickSigmaPrior, est)
        var rng = Rng(seed: seed), samples: [Mat3] = []
        for _ in 0..<nSamples {
            let noisy = img.map { Pt($0.x + rng.normal() * sigma, $0.y + rng.normal() * sigma) }
            if let h = Homography.fit(src: world, dst: noisy), let hi = h.inverse?.normalized { samples.append(hi) }
        }
        guard !samples.isEmpty else { return nil }
        func median(_ v: [Double]) -> Double { let s = v.sorted(); return s[s.count / 2] }
        H = Hic; Hinv = Hci; self.samples = samples
        wSign = median(img.map { Hic.apply($0).w }) >= 0 ? 1 : -1
        wcSign = median(world.map { Hci.apply($0).w }) >= 0 ? 1 : -1
        residualRMS = (resid.map { $0 * $0 }.reduce(0, +) / Double(resid.count)).squareRoot()
        clickSigma = sigma; nPoints = names.count
    }

    /// true if the pixel is below the floor's horizon (can be a floor point).
    func onGround(_ px: Pt) -> Bool { (H.apply(px).w >= 0 ? 1.0 : -1.0) == wSign }
    func toCourt(_ px: Pt) -> Pt { H.project(px) }
    /// Court -> image, nil beyond the camera's horizon.
    func toImage(_ p: Pt) -> Pt? {
        let (q, w) = Hinv.apply(p)
        return (w >= 0 ? 1.0 : -1.0) == wcSign ? q : nil
    }
}
