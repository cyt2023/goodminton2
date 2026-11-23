import Foundation

/// Landing detection from a single-camera 2D track (port of prototype/landing.py).
///
/// A single camera cannot see height, so a landing is recognised from what happens
/// after it: the shuttle arrives in flight and then stays still on the floor.
/// Rally context: a coherent flight before the rest and a long enough rest after
/// it; otherwise the candidate is a juggle, a drop or a pick-up between points.
struct LandingCandidate: Codable {
    var contactFrame: Int
    var contactTime: Double
    var lastFlightFrame: Int
    var restStart: Int, restEnd: Int
    var contactPx: Pt
    var restPx: Pt
    var velocityPxPerFrame: Pt
    var isRallyLanding: Bool
    var hasFlight: Bool
    var restSeconds: Double
    var reasons: [String]
}

enum LandingDetector {
    struct Params {
        var stillSeconds = 0.3, stillFrac = 0.8, arriveSeconds = 1.5
        var minFlightDets = 5, minFlightDisp = 0.08, minRestSeconds = 0.6
        var maxGapSeconds = 0.15, maxSpeed = 2.5
    }

    static func dist(_ a: Pt, _ b: Pt) -> Double { (a - b).norm }

    /// Chains of detections a single moving shuttle could form.
    static func flightChains(_ xy: [Pt?], fps: Double, width: Double, p: Params) -> [[Int]] {
        let idx = xy.indices.filter { xy[$0] != nil }
        guard let first = idx.first else { return [] }
        let maxGap = max(1, Int((p.maxGapSeconds * fps).rounded()))
        let vMax = p.maxSpeed * width / fps
        var chains: [[Int]] = [], cur = [first]
        for (a, b) in zip(idx.dropLast(), idx.dropFirst()) {
            if b - a <= maxGap && dist(xy[b]!, xy[a]!) <= vMax * Double(b - a) { cur.append(b) } else { chains.append(cur); cur = [b] }
        }
        chains.append(cur)
        return chains
    }

    static func detect(_ xy: [Pt?], fps: Double, width: Double, p: Params = Params()) -> [LandingCandidate] {
        let n = xy.count
        let k = max(3, Int((p.stillSeconds * fps).rounded()))
        let rStill = max(3.0, 0.004 * width), rFar = 8 * rStill
        var still = [Bool](repeating: false, count: n), rest = [Pt?](repeating: nil, count: n)
        if n >= k {
            for s in 0...(n - k) {
                let w = (s..<(s + k)).compactMap { xy[$0] }
                guard Double(w.count) >= p.stillFrac * Double(k) else { continue }
                let med = Pt(median(w.map(\.x)), median(w.map(\.y)))
                if w.allSatisfy({ dist($0, med) <= rStill }) { still[s] = true; rest[s] = med }
            }
        }
        let chains = flightChains(xy, fps: fps, width: width, p: p)
        let back = Int((p.arriveSeconds * fps).rounded())
        var out: [LandingCandidate] = []
        var s = 0
        while s < n {
            guard still[s], let pRest = rest[s] else { s += 1; continue }
            var e = s
            while e + 1 < n, still[e + 1], let r = rest[e + 1], dist(r, pRest) <= rStill { e += 1 }
            let lo = max(0, s - back)
            let far = (lo..<s).filter { xy[$0] != nil && dist(xy[$0]!, pRest) > rFar }
            let lastEnd = out.last?.restEnd ?? -1
            if let lastFar = far.last, lastFar > lastEnd {
                let c = context(xy, fps: fps, width: width, s: s, e: e + k - 1, pRest: pRest, rStill: rStill, rFar: rFar,
                                chains: chains, fallbackFar: lastFar, p: p)
                // The same landing again: the rest was briefly broken (occlusion, a stray detection
                // elsewhere) but the shuttle arrived with the same flight, or with none, at the same spot.
                if let prev = out.last,
                   (c.hasFlight && prev.hasFlight && c.lastFlightFrame == prev.lastFlightFrame) ||
                   (!c.hasFlight && dist(pRest, prev.restPx) <= rFar) {
                    out[out.count - 1].restEnd = e + k - 1
                } else {
                    out.append(c)
                }
            }
            s = e + k
        }
        return out
    }

    private static func context(_ xy: [Pt?], fps: Double, width: Double, s: Int, e: Int, pRest: Pt, rStill: Double, rFar: Double,
                                chains: [[Int]], fallbackFar: Int, p: Params) -> LandingCandidate {
        let back = Int((p.arriveSeconds * fps).rounded())
        var flight: (start: Int, end: Int, v: Pt)? = nil
        for ch in chains.reversed() {
            let pre = ch.filter { $0 < s && $0 >= s - back }
            if pre.isEmpty { continue }
            let moving = pre.filter { dist(xy[$0]!, pRest) > rFar }
            if moving.count < p.minFlightDets { continue }
            let end = moving.last!
            let disp = moving.map { dist(xy[$0]!, xy[end]!) }.max() ?? 0
            let tail = Array(moving.suffix(3))
            let v = (xy[tail.last!]! - xy[tail.first!]!) * (1.0 / Double(max(1, tail.last! - tail.first!)))
            let reach = v.norm * Double(s - end) * 1.5 + rFar
            if disp >= p.minFlightDisp * width && dist(xy[end]!, pRest) <= reach {
                flight = (moving.first!, end, v)
                break
            }
        }
        let after = flight?.end ?? (s - 1)
        let vis = ((after + 1)..<xy.count).filter { xy[$0] != nil }
        let contact = vis.first { dist(xy[$0]!, pRest) <= 2 * rStill } ?? s
        var lastNear = contact, away = 0
        for f in vis where f >= contact {
            if dist(xy[f]!, pRest) <= 2 * rStill { lastNear = f; away = 0 } else { away += 1; if away >= 2 { break } }
        }
        let restS = Double(lastNear - contact) / fps
        var reasons: [String] = []
        if flight == nil { reasons.append("没有找到落地前的连续飞行轨迹") }
        if restS < p.minRestSeconds { reasons.append(String(format: "落地后只静止了 %.2f s，像颠球或马上被捡起", restS)) }
        let lastFlight = flight?.end ?? fallbackFar
        let v = flight?.v ?? ((xy[contact] ?? pRest) - xy[lastFlight]!) * (1.0 / Double(max(1, contact - lastFlight)))
        return LandingCandidate(contactFrame: contact, contactTime: Double(contact) / fps, lastFlightFrame: lastFlight,
                                restStart: s, restEnd: e, contactPx: xy[contact] ?? pRest, restPx: pRest, velocityPxPerFrame: v,
                                isRallyLanding: reasons.isEmpty, hasFlight: flight != nil, restSeconds: restS, reasons: reasons)
    }

    static func median(_ v: [Double]) -> Double {
        let s = v.sorted(), n = s.count
        return n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
    }
}
