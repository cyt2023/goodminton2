import Foundation

/// Automatic court calibration from one frame: find painted lines, fit the badminton
/// court model, and return the line intersections as calibration points. The user
/// then checks and corrects them; nothing here is trusted blindly.
///
/// 1. Line pixels: thin stripes that differ from the floor on both sides (any colour).
/// 2. Hough transform -> candidate straight lines, split into "across" (near horizontal
///    in the picture, e.g. baselines and service lines seen from behind a baseline)
///    and "along" (sidelines, centre line).
/// 3. Hypotheses: 2 across x 2 along lines, assigned to 2 court y-lines x 2 court x-lines;
///    the homography from the 4 intersections is scored by how many points of the
///    projected court model land on line pixels.
/// 4. Refinement: projected model lines are snapped to the centre of the nearby painted
///    line and the homography is refitted on hundreds of points.
enum CourtDetector {
    struct Result {
        var points: [String: Pt]      // visible keypoints, original pixels
        var courtToImage: Mat3        // original pixels
        var score: Double             // fraction of visible model samples on painted lines
        var visibleSamples: Int
    }

    static let workWidth = 960

    /// rgba: 4 bytes per pixel, any byte order where bytes 0...2 are colour channels.
    static func detect(pixels: UnsafePointer<UInt8>, width: Int, height: Int, rowBytes: Int) -> Result? {
        // Downscale to ~960 px wide (area average), as RGB floats.
        let s = min(1.0, Double(workWidth) / Double(width))
        let w = max(1, Int(Double(width) * s)), h = max(1, Int(Double(height) * s))
        var img = [Float](repeating: 0, count: w * h * 3)
        let fx = Double(width) / Double(w), fy = Double(height) / Double(h)
        for y in 0..<h {
            let y0 = Int(Double(y) * fy), y1 = max(y0 + 1, Int(Double(y + 1) * fy))
            for x in 0..<w {
                let x0 = Int(Double(x) * fx), x1 = max(x0 + 1, Int(Double(x + 1) * fx))
                var r: Float = 0, g: Float = 0, b: Float = 0, n: Float = 0
                var yy = y0
                while yy < min(y1, height) {
                    let row = pixels + yy * rowBytes
                    var xx = x0
                    while xx < min(x1, width) { let p = row + xx * 4; r += Float(p[0]); g += Float(p[1]); b += Float(p[2]); n += 1; xx += 1 }
                    yy += 1
                }
                let o = (y * w + x) * 3
                img[o] = r / n; img[o + 1] = g / n; img[o + 2] = b / n
            }
        }
        let mask = lineMask(img, w, h)
        guard let fit = fitModel(mask, w, h) else { return nil }
        // Back to original pixels.
        let S = Mat3([1 / s, 0, 0, 0, 1 / s, 0, 0, 0, 1])
        let H = (S * fit.H).normalized
        var pts: [String: Pt] = [:]
        for kp in Court.keypoints {
            let (q, wq) = H.apply(kp.xy)
            guard (wq >= 0 ? 1.0 : -1.0) == fit.wSign, q.x >= 0, q.y >= 0, q.x < Double(width), q.y < Double(height) else { continue }
            pts[kp.name] = q
        }
        guard pts.count >= 4 else { return nil }
        return Result(points: pts, courtToImage: H, score: fit.score, visibleSamples: fit.visible)
    }

    /// Work image: line mask at ~960 px wide, and the scale from original pixels.
    struct Prepared { var mask: [Bool]; var w: Int; var h: Int; var scale: Double; var width: Int; var height: Int }

    static func prepare(pixels: UnsafePointer<UInt8>, width: Int, height: Int, rowBytes: Int) -> Prepared {
        let s = min(1.0, Double(workWidth) / Double(width))
        let w = max(1, Int(Double(width) * s)), h = max(1, Int(Double(height) * s))
        var img = [Float](repeating: 0, count: w * h * 3)
        let fx = Double(width) / Double(w), fy = Double(height) / Double(h)
        for y in 0..<h {
            let y0 = Int(Double(y) * fy), y1 = max(y0 + 1, Int(Double(y + 1) * fy))
            for x in 0..<w {
                let x0 = Int(Double(x) * fx), x1 = max(x0 + 1, Int(Double(x + 1) * fx))
                var r: Float = 0, g: Float = 0, b: Float = 0, n: Float = 0
                var yy = y0
                while yy < min(y1, height) {
                    let row = pixels + yy * rowBytes
                    var xx = x0
                    while xx < min(x1, width) { let p = row + xx * 4; r += Float(p[0]); g += Float(p[1]); b += Float(p[2]); n += 1; xx += 1 }
                    yy += 1
                }
                let o = (y * w + x) * 3
                img[o] = r / n; img[o + 1] = g / n; img[o + 2] = b / n
            }
        }
        return Prepared(mask: lineMask(img, w, h), w: w, h: h, scale: s, width: width, height: height)
    }

    /// From a rough calibration (e.g. four tapped corners), snap the court model onto the
    /// painted lines and return every visible keypoint. Nil if the lines are not found.
    static func refine(points: [String: Pt], _ prep: Prepared) -> Result? {
        let names = points.keys.filter { Court.keypointByName[$0] != nil }
        guard names.count >= 4 else { return nil }
        let s = prep.scale
        guard var H = Homography.fit(src: names.map { Court.keypointByName[$0]! }, dst: names.map { points[$0]! * s }) else { return nil }
        let wSign: Double = H.apply(Pt(Court.width / 2, Court.length / 2)).w >= 0 ? 1 : -1
        var dil = prep.mask
        for y in 1..<(prep.h - 1) { for x in 1..<(prep.w - 1) where prep.mask[y * prep.w + x] {
            for dy in -1...1 { for dx in -1...1 { dil[(y + dy) * prep.w + x + dx] = true } }
        } }
        let samples = modelSamples(step: 0.15)
        let H0 = H
        var Hs = H
        for radius in [10.0, 6.0, 4.0, 3.0] {
            if let Hr = snap(Hs, prep.mask, prep.w, prep.h, wSign: wSign, radius: radius), plausible(Hr, prep.w, prep.h, wSign: wSign) { Hs = Hr }
        }
        // Keep the snapped fit only if it explains more painted line and stays close to the
        // user's taps (12 px at work scale); otherwise the taps are used as they are.
        let (_, on0, vis0) = score(H0, samples, prep.mask, dil, prep.w, prep.h, wSign: wSign)
        let (_, on1, vis1) = score(Hs, samples, prep.mask, dil, prep.w, prep.h, wSign: wSign)
        let drift = names.map { (Hs.project(Court.keypointByName[$0]!) - points[$0]! * s).norm }.max() ?? 0
        let r0 = vis0 > 0 ? Double(on0) / Double(vis0) : 0, r1 = vis1 > 0 ? Double(on1) / Double(vis1) : 0
        if r1 > r0 + 0.05 && drift < 12 { H = Hs }
        let (_, on, vis) = score(H, samples, prep.mask, dil, prep.w, prep.h, wSign: wSign)
        let Ho = (Mat3([1 / s, 0, 0, 0, 1 / s, 0, 0, 0, 1]) * H).normalized
        var pts: [String: Pt] = [:]
        for kp in Court.keypoints {
            let (q, wq) = Ho.apply(kp.xy)
            guard (wq >= 0 ? 1.0 : -1.0) == wSign, q.x >= 0, q.y >= 0, q.x < Double(prep.width), q.y < Double(prep.height) else { continue }
            pts[kp.name] = q
        }
        return Result(points: pts, courtToImage: Ho, score: vis > 0 ? Double(on) / Double(vis) : 0, visibleSamples: vis)
    }

    // MARK: line pixels

    /// Thin stripes whose background on both sides looks like the floor. The floor
    /// colour is taken from the bottom quarter of the picture (the camera stands behind
    /// a baseline, so the floor fills the bottom), which removes curtains, ceilings, lights.
    static func lineMask(_ img: [Float], _ w: Int, _ h: Int) -> [Bool] {
        func px(_ x: Int, _ y: Int) -> (Float, Float, Float) { let o = (y * w + x) * 3; return (img[o], img[o + 1], img[o + 2]) }
        func d(_ a: (Float, Float, Float), _ b: (Float, Float, Float)) -> Float { (abs(a.0 - b.0) + abs(a.1 - b.1) + abs(a.2 - b.2)) / 3 }
        // Floor colours: a few clusters from the bottom quarter (multi-sport halls have patches).
        var samples: [(Float, Float, Float)] = []
        for y in stride(from: h * 3 / 4, to: h, by: 4) { for x in stride(from: 0, to: w, by: 4) { samples.append(px(x, y)) } }
        func median(_ v: [Float]) -> Float { let s = v.sorted(); return s.isEmpty ? 0 : s[s.count / 2] }
        // Colour without brightness (chromaticity), so shadows on the floor still count
        // as floor while a grey curtain or a white line does not.
        func chroma(_ c: (Float, Float, Float)) -> (Float, Float, Float) {
            let sum = max(c.0 + c.1 + c.2, 1); return (c.0 / sum, c.1 / sum, sum / 3)
        }
        func same(_ a: (Float, Float, Float), _ f: (Float, Float, Float)) -> Bool {
            let ca = chroma(a), cf = chroma(f)
            return abs(ca.0 - cf.0) + abs(ca.1 - cf.1) < 0.045 && ca.2 > 0.55 * cf.2 && ca.2 < 1.5 * cf.2
        }
        var floors: [(Float, Float, Float)] = []
        var rest = samples
        for _ in 0..<3 {
            let f = (median(rest.map(\.0)), median(rest.map(\.1)), median(rest.map(\.2)))
            let members = rest.filter { same($0, f) }.count
            // A floor colour covers a good part of the bottom; lines and people do not.
            guard members >= samples.count / 4 else { break }
            floors.append(f)
            rest = rest.filter { !same($0, f) }
            if rest.count < samples.count / 4 { break }
        }
        func isFloor(_ c: (Float, Float, Float)) -> Bool { floors.contains { same(c, $0) } }
        var m = [Bool](repeating: false, count: w * h)
        let t: Float = 12
        for y in 0..<h {
            for x in 0..<w {
                let c = px(x, y)
                if isFloor(c) { continue }
                var hit = false
                for dd in [2, 4, 7] where !hit {
                    if x >= dd && x + dd < w {
                        let l = px(x - dd, y), r = px(x + dd, y)
                        if d(c, l) > t && d(c, r) > t && isFloor(l) && isFloor(r) { hit = true }
                    }
                    if !hit && y >= dd && y + dd < h {
                        let u = px(x, y - dd), b = px(x, y + dd)
                        if d(c, u) > t && d(c, b) > t && isFloor(u) && isFloor(b) { hit = true }
                    }
                }
                m[y * w + x] = hit
            }
        }
        return m
    }

    // MARK: lines

    struct Line { var theta: Double; var rho: Double; var votes: Int
        /// y at a given x (for near-horizontal lines) / x at a given y.
        func y(at x: Double) -> Double { (rho - x * cos(theta)) / sin(theta) }
        func x(at y: Double) -> Double { (rho - y * sin(theta)) / cos(theta) }
    }

    /// Hough peaks, each checked for real extent: line pixels within 1.5 px of the line
    /// must occupy enough distinct 6 px bins along it (a blob of pixels does not).
    static func hough(_ mask: [Bool], _ w: Int, _ h: Int, maxLines: Int = 40) -> [Line] {
        let nT = 180, diag = Int((Double(w * w + h * h)).squareRoot()) + 1
        let nR = 2 * diag + 1
        var acc = [Int32](repeating: 0, count: nT * nR)
        let cs = (0..<nT).map { cos(Double($0) * .pi / Double(nT)) }, sn = (0..<nT).map { sin(Double($0) * .pi / Double(nT)) }
        var pts: [(Double, Double)] = []
        for y in 0..<h { for x in 0..<w where mask[y * w + x] {
            pts.append((Double(x), Double(y)))
            for t in 0..<nT {
                let r = Int((Double(x) * cs[t] + Double(y) * sn[t]).rounded()) + diag
                acc[t * nR + r] += 1
            }
        } }
        var lines: [Line] = []
        for _ in 0..<(maxLines * 3) {
            var best = 0
            for i in 0..<acc.count where acc[i] > acc[best] { best = i }
            if acc[best] < 12 { break }
            let t = best / nR, r = best % nR
            let th = Double(t) * .pi / Double(nT), rho = Double(r - diag)
            for dt in -3...3 { for dr in -6...6 {
                let tt = t + dt
                let (t2, r2) = tt < 0 ? (tt + nT, 2 * diag - r - dr) : (tt >= nT ? (tt - nT, 2 * diag - r - dr) : (tt, r + dr))
                if r2 >= 0 && r2 < nR { acc[t2 * nR + r2] = 0 }
            } }
            // Coverage along the line.
            var bins = Set<Int>()
            for (x, y) in pts where abs(x * cs[t] + y * sn[t] - rho) <= 1.5 {
                bins.insert(Int(((-x * sn[t] + y * cs[t]) / 6).rounded(.down)))
            }
            if bins.count >= 8 { lines.append(Line(theta: th, rho: rho, votes: bins.count)) }
            if lines.count >= maxLines { break }
        }
        return lines.sorted { $0.votes > $1.votes }
    }

    /// Merge detections of the same thick painted line (within 2.5 degrees and 10 px).
    static func dedupe(_ lines: [Line]) -> [Line] {
        var out: [Line] = []
        for l in lines {
            let dup = out.contains { o in
                var dt = abs(o.theta - l.theta), dr = abs(o.rho - l.rho)
                if dt > .pi / 2 { dt = .pi - dt; dr = abs(o.rho + l.rho) }
                return dt < 2.5 * .pi / 180 && dr < 10
            }
            if !dup { out.append(l) }
        }
        return out
    }

    /// Lines along the court all meet at one vanishing point; find the point most lines
    /// pass near and call those "along". The others (across the court) are near parallel.
    static func classify(_ lines: [Line], _ w: Int, _ h: Int) -> (across: [Line], along: [Line]) {
        let top = Array(lines.prefix(30))
        func dist(_ p: Pt, _ l: Line) -> Double { abs(p.x * cos(l.theta) + p.y * sin(l.theta) - l.rho) }
        var bestVP: Pt? = nil, bestN = 0
        // Camera behind a baseline looking down the court: the vanishing point of the
        // along-court lines lies roughly above the middle of the picture, near the horizon.
        let tol = 6.0
        for i in 0..<top.count { for j in (i + 1)..<top.count {
            guard let p = intersect(top[i], top[j]), abs(p.x - Double(w) / 2) < Double(w),
                  p.y > -1.5 * Double(h), p.y < 0.75 * Double(h) else { continue }
            let through = top.filter { dist(p, $0) < tol }
            guard through.count >= 2 else { continue }
            let n = through.reduce(0) { $0 + $1.votes }
            if n > bestN { bestN = n; bestVP = p }
        }}
        guard var vp = bestVP else { return ([], []) }
        // Least-squares vanishing point of the lines through it, then a looser selection.
        for _ in 0..<2 {
            let sel = lines.filter { dist(vp, $0) < 12 }
            var a11 = 0.0, a12 = 0.0, a22 = 0.0, b1 = 0.0, b2 = 0.0
            for l in sel { let c = cos(l.theta), s = sin(l.theta), wt = Double(l.votes)
                a11 += wt * c * c; a12 += wt * c * s; a22 += wt * s * s; b1 += wt * c * l.rho; b2 += wt * s * l.rho }
            let det = a11 * a22 - a12 * a12
            if sel.count >= 2, abs(det) > 1e-9 { vp = Pt((b1 * a22 - b2 * a12) / det, (a11 * b2 - a12 * b1) / det) }
        }
        let bottom = Double(h), mid = Double(w) / 2
        var along: [Line] = []
        for l in lines where dist(vp, l) < 14 {
            // one entry per painted line: where it crosses the bottom of the picture
            if along.contains(where: { abs($0.x(at: bottom) - l.x(at: bottom)) < 18 }) { continue }
            along.append(l)
        }
        var across: [Line] = []
        for l in lines where dist(vp, l) >= 14 && abs(l.theta * 180 / .pi - 90) < 45 {
            if across.contains(where: { abs($0.y(at: mid) - l.y(at: mid)) < 8 }) { continue }
            across.append(l)
        }
        return (across, along)
    }

    static func intersect(_ a: Line, _ b: Line) -> Pt? {
        let det = cos(a.theta) * sin(b.theta) - sin(a.theta) * cos(b.theta)
        guard abs(det) > 1e-6 else { return nil }
        return Pt((a.rho * sin(b.theta) - b.rho * sin(a.theta)) / det, (b.rho * cos(a.theta) - a.rho * cos(b.theta)) / det)
    }

    // MARK: model fit

    struct Fit { var H: Mat3; var score: Double; var visible: Int; var wSign: Double }

    /// Points along every painted line of the model, spacing in metres.
    static func modelSamples(step: Double) -> [Pt] {
        var out: [Pt] = []
        for (a, b) in Court.lineSegments() {
            let n = max(1, Int(((b - a).norm / step).rounded()))
            for i in 0...n { out.append(a + (b - a) * (Double(i) / Double(n))) }
        }
        return out
    }

    /// Counts distinct pixels (in 2 px cells), so a degenerate fit that piles the whole
    /// court onto one painted line cannot collect points.
    static func score(_ H: Mat3, _ samples: [Pt], _ mask: [Bool], _ dil: [Bool], _ w: Int, _ h: Int, wSign: Double) -> (Double, Int, Int) {
        var seen = Set<Int>(), on = 0, vis = 0, outside = 0
        for p in samples {
            let (q, wq) = H.apply(p)
            // The player is told to frame the whole court, so model lines outside the
            // picture count against a hypothesis.
            guard (wq >= 0 ? 1.0 : -1.0) == wSign, q.x >= 0, q.y >= 0, q.x < Double(w - 1), q.y < Double(h - 1) else { outside += 1; continue }
            let cell = (Int(q.y) / 2) * w + Int(q.x) / 2
            if !seen.insert(cell).inserted { continue }
            vis += 1
            if dil[Int(q.y) * w + Int(q.x)] { on += 1 }
        }
        return (Double(on) - 0.5 * Double(vis - on) - 0.3 * Double(outside) * Double(vis) / Double(max(1, samples.count - outside)), on, vis)
    }

    /// The doubles court (or the part in front of the camera) must project to a convex,
    /// reasonably large quadrilateral.
    static func plausible(_ H: Mat3, _ w: Int, _ h: Int, wSign: Double) -> Bool {
        let corners = [Pt(0, 0), Pt(Court.width, 0), Pt(Court.width, Court.netY), Pt(0, Court.netY)]
        var q: [Pt] = []
        for c in corners {
            let (p, wp) = H.apply(c)
            guard (wp >= 0 ? 1.0 : -1.0) == wSign, p.x.isFinite, p.y.isFinite else { return false }
            q.append(p)
        }
        var area = 0.0, sign = 0.0
        for i in 0..<4 {
            let a = q[i], b = q[(i + 1) % 4], c = q[(i + 2) % 4]
            let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            if sign == 0 { sign = cross > 0 ? 1 : -1 } else if cross * sign <= 0 { return false }
            area += a.x * b.y - b.x * a.y
        }
        return abs(area) / 2 > 0.03 * Double(w * h)
    }

    static func fitModel(_ mask: [Bool], _ w: Int, _ h: Int) -> Fit? {
        // Dilate by 1 px for scoring.
        var dil = mask
        for y in 1..<(h - 1) { for x in 1..<(w - 1) where mask[y * w + x] {
            for dy in -1...1 { for dx in -1...1 { dil[(y + dy) * w + x + dx] = true } }
        } }
        let (across0, along0) = classify(dedupe(hough(mask, w, h)), w, h)
        let across = Array(across0.prefix(8)), along = Array(along0.prefix(8))
        if ProcessInfo.processInfo.environment["COURT_DEBUG"] != nil {
            let f = { (l: Line) in String(format: "(%.0f°, %.0f, x@bottom %.0f, y@mid %.0f, cov %d)", l.theta * 180 / .pi, l.rho, l.x(at: Double(h)), l.y(at: Double(w) / 2), l.votes) }
            print("across:", across.map(f).joined(separator: " ")); print("along:", along.map(f).joined(separator: " "))
        }
        guard across.count >= 2, along.count >= 2 else { return nil }
        let quick = modelSamples(step: 1.2), full = modelSamples(step: 0.15)
        let ys = Court.yCentres.map(\.1), xs = Court.xCentres.map(\.1)
        let cx = Double(w) / 2, bottom = Double(h)
        var cands: [(Double, Mat3, Double)] = []
        for i in 0..<across.count { for j in 0..<across.count where i != j {
            // i is nearer to the camera (lower in the picture) than j
            guard across[i].y(at: cx) > across[j].y(at: cx) + 4 else { continue }
            for k in 0..<along.count { for l in 0..<along.count where k != l {
                // k is left of l at the bottom of the picture
                guard along[k].x(at: bottom) < along[l].x(at: bottom) - 4 else { continue }
                guard let p00 = intersect(across[i], along[k]), let p01 = intersect(across[i], along[l]),
                      let p10 = intersect(across[j], along[k]), let p11 = intersect(across[j], along[l]) else { continue }
                // Intersections must be roughly inside an extended picture.
                if [p00, p01, p10, p11].contains(where: { abs($0.x - cx) > 3 * Double(w) || $0.y < -2 * Double(h) || $0.y > 3 * Double(h) }) { continue }
                let minGap = 0.02 * Double(w)
                if (p00 - p01).norm < minGap || (p10 - p11).norm < minGap || (p00 - p10).norm < minGap || (p01 - p11).norm < minGap { continue }
                for a in 0..<ys.count { for b in (a + 1)..<ys.count {
                    for c in 0..<xs.count { for d in (c + 1)..<xs.count {
                        let src = [Pt(xs[c], ys[a]), Pt(xs[d], ys[a]), Pt(xs[c], ys[b]), Pt(xs[d], ys[b])]
                        guard let H = Homography.fit(src: src, dst: [p00, p01, p10, p11], refine: false) else { continue }
                        // The camera must be outside the court on the near side: the court
                        // centre must project in front of the camera.
                        let wSign: Double = H.apply(Pt(Court.width / 2, Court.length / 2)).w >= 0 ? 1 : -1
                        guard plausible(H, w, h, wSign: wSign) else { continue }
                        let (sc, _, vis) = score(H, quick, mask, dil, w, h, wSign: wSign)
                        if vis < 12 { continue }
                        cands.append((sc, H, wSign))
                    } }
                } }
            } }
        } }
        cands.sort { $0.0 > $1.0 }
        var best: Fit? = nil, bestScore = -Double.infinity
        for (_, H0, wSign) in cands.prefix(25) {
            var H = H0
            for radius in [6.0, 4.0, 3.0] { if let Hr = snap(H, mask, w, h, wSign: wSign, radius: radius) { H = Hr } }
            if !plausible(H, w, h, wSign: wSign) { H = H0 }
            let (sc, on, vis) = score(H, full, mask, dil, w, h, wSign: wSign)
            if vis < 40 { continue }
            if sc > bestScore {
                bestScore = sc
                best = Fit(H: H, score: Double(on) / Double(vis), visible: vis, wSign: wSign)
            }
        }
        return best
    }

    /// Move each projected model sample to the centre of the painted line along the
    /// picture normal of the model line, then refit the homography.
    static func snap(_ H: Mat3, _ mask: [Bool], _ w: Int, _ h: Int, wSign: Double, radius: Double) -> Mat3? {
        var src: [Pt] = [], dst: [Pt] = []
        for (a, b) in Court.lineSegments() {
            let len = (b - a).norm, n = max(2, Int(len / 0.15))
            for i in 0...n {
                let p = a + (b - a) * (Double(i) / Double(n))
                let p2 = a + (b - a) * (min(1, Double(i) / Double(n) + 0.02))
                let p1 = a + (b - a) * (max(0, Double(i) / Double(n) - 0.02))
                let (q, wq) = H.apply(p)
                guard (wq >= 0 ? 1.0 : -1.0) == wSign else { continue }
                let d = H.project(p2) - H.project(p1)
                guard d.norm > 1e-6 else { continue }
                let nrm = Pt(-d.y / d.norm, d.x / d.norm)
                var hits: [Double] = []
                var t = -radius
                while t <= radius {
                    let x = Int((q.x + nrm.x * t).rounded()), y = Int((q.y + nrm.y * t).rounded())
                    if x >= 0, y >= 0, x < w, y < h, mask[y * w + x] { hits.append(t) }
                    t += 0.5
                }
                guard hits.count >= 2 else { continue }
                let c = hits.reduce(0, +) / Double(hits.count)
                src.append(p); dst.append(q + nrm * c)
            }
        }
        guard src.count >= 12 else { return nil }
        return Homography.fit(src: src, dst: dst, refine: false)
    }
}
