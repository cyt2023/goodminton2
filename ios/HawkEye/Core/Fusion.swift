import Foundation

/// One phone's view of a landing (kept next to the combined result).
struct LandingView: Codable {
    var side: String               // "A" (this phone / host) or "B" (the other end)
    var result: LandingResult      // in the host's court coordinates
    var snapshotFile: String?
}

/// Combining the two phones' estimates of the same landing.
///
/// Each phone stands behind one baseline and calibrates with its own end as "near".
/// The guest's results are turned 180 degrees into the host's court coordinates, then
/// the two Gaussian estimates are combined by inverse-covariance weighting, so the
/// phone that sees the landing better (usually the nearer one) dominates. P(in) and the
/// call are recomputed from the combined Gaussian.
enum Fusion {
    /// Court coordinates as seen from the other end: (x, y) -> (W - x, L - y).
    static func rotate(_ r: LandingResult, mode: Mode) -> LandingResult {
        var o = r
        o.courtXY = Pt(Court.width - r.courtXY.x, Court.length - r.courtXY.y)
        let (d, edge) = Court.signedDistance(o.courtXY, mode)
        o.nearestEdge = edge
        o.signedDistance = d + (r.signedDistance - Court.signedDistance(r.courtXY, mode).d)   // keep the contact-radius offset
        return o                                     // a 180 degree turn leaves the covariance unchanged
    }

    static func inv2(_ c: [Double]) -> [Double]? {
        let det = c[0] * c[3] - c[1] * c[2]
        guard abs(det) > 1e-18, det.isFinite else { return nil }
        return [c[3] / det, -c[1] / det, -c[2] / det, c[0] / det]
    }

    /// In/out call from a Gaussian landing estimate (Monte Carlo, like Uncertainty.locate).
    static func fromGaussian(xy: Pt, cov: [Double], mode: Mode, contactRadius: Double, alpha: Double, n: Int = 4000, seed: UInt64 = 3) -> LandingResult {
        let l11 = max(cov[0], 1e-12).squareRoot(), l21 = cov[2] / l11, l22 = max(cov[3] - l21 * l21, 0).squareRoot()
        var rng = Rng(seed: seed)
        var inside = 0
        var ds: [Double] = []
        ds.reserveCapacity(n)
        for _ in 0..<n {
            let z1 = rng.normal(), z2 = rng.normal()
            let p = Pt(xy.x + l11 * z1, xy.y + l21 * z1 + l22 * z2)
            if Court.isInside(p, mode, contactRadius: contactRadius) { inside += 1 }
            ds.append(Court.signedDistance(p, mode).d)
        }
        let m = ds.reduce(0, +) / Double(n)
        let sd = (ds.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(n - 1)).squareRoot()
        let pIn = Double(inside) / Double(n)
        let call: Call = pIn >= 1 - alpha ? .IN : (pIn <= alpha ? .OUT : .ABSTAIN)
        let (d, edge) = Court.signedDistance(xy, mode)
        return LandingResult(courtXY: xy, cov: cov, signedDistance: d + contactRadius, distanceSigma: sd, nearestEdge: edge, pIn: pIn,
                             call: call, abstainReason: call == .ABSTAIN ? String(format: "两机合并后 P(界内)=%.2f，离线太近", pIn) : nil,
                             fractionAboveHorizon: 0)
    }

    /// Combined estimate. A view whose samples crossed its horizon is not Gaussian and is
    /// left out if the other view is fine.
    static func combine(_ a: LandingResult, _ b: LandingResult, mode: Mode, contactRadius: Double, alpha: Double) -> LandingResult {
        let aBad = a.fractionAboveHorizon > 0.01, bBad = b.fractionAboveHorizon > 0.01
        if aBad && !bBad { return b }
        if bBad && !aBad { return a }
        let ridge = 1e-8
        guard let ia = inv2([a.cov[0] + ridge, a.cov[1], a.cov[2], a.cov[3] + ridge]),
              let ib = inv2([b.cov[0] + ridge, b.cov[1], b.cov[2], b.cov[3] + ridge]),
              let p = inv2(zip(ia, ib).map { $0 + $1 }) else { return a.distanceSigma <= b.distanceSigma ? a : b }
        let va = Pt(ia[0] * a.courtXY.x + ia[1] * a.courtXY.y, ia[2] * a.courtXY.x + ia[3] * a.courtXY.y)
        let vb = Pt(ib[0] * b.courtXY.x + ib[1] * b.courtXY.y, ib[2] * b.courtXY.x + ib[3] * b.courtXY.y)
        let s = va + vb
        let xy = Pt(p[0] * s.x + p[1] * s.y, p[2] * s.x + p[3] * s.y)
        return fromGaussian(xy: xy, cov: p, mode: mode, contactRadius: contactRadius, alpha: alpha)
    }
}

/// Pairs up the landings reported by the two phones (host clock) and emits each landing
/// once: combined when both saw it, otherwise from the one phone that did.
final class LandingFusion {
    struct Item {
        var landing: Landing
        var time: Double            // contact time, host clock (s)
        var side: String            // "A" or "B"
        var snapshotFile: String?
        var arrived: Double         // host clock when it arrived here
    }
    var window = 0.35, maxDistance = 1.5, wait = 2.0, keepEmitted = 15.0
    let mode: Mode
    let contactRadius: Double, alpha: Double
    private var pending: [Item] = []
    /// Landings already shown from one phone only: a late partner updates them in place.
    private var emitted: [(item: Item, landing: Landing)] = []

    init(mode: Mode, contactRadius: Double, alpha: Double) { self.mode = mode; self.contactRadius = contactRadius; self.alpha = alpha }

    private func pairs(_ p: Item, _ item: Item) -> Bool {
        p.side != item.side && abs(p.time - item.time) <= window && (p.landing.result.courtXY - item.landing.result.courtXY).norm <= maxDistance
    }

    /// Add one phone's landing; returns landings that are now final. A returned landing
    /// with the same id as one returned earlier replaces it (late partner).
    func add(_ item: Item) -> [Landing] {
        emitted.removeAll { item.arrived - $0.item.arrived > keepEmitted }
        if let i = pending.firstIndex(where: { pairs($0, item) }) {
            let other = pending.remove(at: i)
            return [merge(item, other)]
        }
        if let i = emitted.firstIndex(where: { pairs($0.item, item) }) {
            let e = emitted.remove(at: i)
            var l = merge(item, e.item)
            l.candidate = e.landing.candidate        // keep the id of the landing already shown
            return [l]
        }
        pending.append(item)
        return []
    }

    /// Landings whose partner did not arrive in time (call regularly).
    func flush(now: Double, all: Bool = false) -> [Landing] {
        let due = pending.filter { all || now - $0.arrived >= wait }
        pending.removeAll { p in due.contains { $0.side == p.side && $0.time == p.time } }
        let out = due.map { single($0) }
        for (it, l) in zip(due, out) { emitted.append((it, l)) }
        return out
    }

    private func single(_ it: Item) -> Landing {
        var l = it.landing
        l.views = [LandingView(side: it.side, result: it.landing.result, snapshotFile: it.snapshotFile)]
        l.snapshotFile = it.snapshotFile
        return l
    }

    private func merge(_ x: Item, _ y: Item) -> Landing {
        let (a, b) = x.side == "A" ? (x, y) : (y, x)
        var l = a.landing
        l.result = Fusion.combine(a.landing.result, b.landing.result, mode: mode, contactRadius: contactRadius, alpha: alpha)
        l.views = [LandingView(side: "A", result: a.landing.result, snapshotFile: a.snapshotFile),
                   LandingView(side: "B", result: b.landing.result, snapshotFile: b.snapshotFile)]
        l.snapshotFile = a.snapshotFile ?? b.snapshotFile
        return l
    }
}
