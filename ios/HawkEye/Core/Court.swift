import Foundation

/// BWF court geometry in metres. Origin at the near-left outer corner, x across
/// (0...6.10), y along the court (0...13.40); "near" is y = 0. Keypoints are the
/// centres of line intersections; the in/out boundary is the OUTER edge of the
/// boundary lines (a shuttle touching the line is in).
enum Court {
    static let width = 6.10
    static let length = 13.40
    static let netY = 6.70
    static let line = 0.04
    static let singlesInset = 0.46
    static let longServiceInset = 0.76
    static let shortServiceFromNet = 1.98

    static let xCentres: [(String, Double)] = [
        ("doubles_left", line / 2), ("singles_left", singlesInset - line / 2), ("centre", width / 2),
        ("singles_right", width - singlesInset + line / 2), ("doubles_right", width - line / 2)]
    static let yCentres: [(String, Double)] = [
        ("back_near", line / 2), ("long_service_near", longServiceInset - line / 2),
        ("short_service_near", netY - shortServiceFromNet - line / 2),
        ("short_service_far", netY + shortServiceFromNet + line / 2),
        ("long_service_far", length - longServiceInset + line / 2), ("back_far", length - line / 2)]

    static let xLabel = ["doubles_left": "左双打边线", "singles_left": "左单打边线", "centre": "中线",
                         "singles_right": "右单打边线", "doubles_right": "右双打边线"]
    static let yLabel = ["back_near": "近端底线", "long_service_near": "近端双打后发球线",
                         "short_service_near": "近端前发球线", "short_service_far": "远端前发球线",
                         "long_service_far": "远端双打后发球线", "back_far": "远端底线"]

    /// All 30 intersections, named "y_name/x_name", in a stable order (near end first).
    static let keypoints: [(name: String, xy: Pt)] = {
        var out: [(String, Pt)] = []
        for (yn, y) in yCentres { for (xn, x) in xCentres { out.append(("\(yn)/\(xn)", Pt(x, y))) } }
        return out
    }()
    static let keypointByName: [String: Pt] = Dictionary(uniqueKeysWithValues: keypoints.map { ($0.name, $0.xy) })

    static func label(_ name: String) -> String {
        let p = name.split(separator: "/").map(String.init)
        return "\(yLabel[p[0]] ?? p[0]) × \(xLabel[p[1]] ?? p[1])"
    }

    /// In-court rectangle for a rally (outer edges of the boundary lines): x0, x1, y0, y1.
    static func boundary(_ mode: Mode) -> (x0: Double, x1: Double, y0: Double, y1: Double) {
        mode == .doubles ? (0, width, 0, length) : (singlesInset, width - singlesInset, 0, length)
    }

    /// Signed distance (m) to the boundary, positive inside, and the nearest edge.
    static func signedDistance(_ p: Pt, _ mode: Mode) -> (d: Double, edge: Edge) {
        let b = boundary(mode)
        let dists: [(Edge, Double)] = [(.left, p.x - b.x0), (.right, b.x1 - p.x), (.near, p.y - b.y0), (.far, b.y1 - p.y)]
        let inside = dists.allSatisfy { $0.1 >= 0 }
        if inside {
            let m = dists.min { $0.1 < $1.1 }!
            return (m.1, m.0)
        }
        // Outside: Euclidean distance to the rectangle, negative.
        let dx = max(b.x0 - p.x, 0, p.x - b.x1), dy = max(b.y0 - p.y, 0, p.y - b.y1)
        let edge = dists.min { $0.1 < $1.1 }!.0
        return (-(dx * dx + dy * dy).squareRoot(), edge)
    }

    static func isInside(_ p: Pt, _ mode: Mode, contactRadius: Double = 0) -> Bool {
        let b = boundary(mode)
        return p.x >= b.x0 - contactRadius && p.x <= b.x1 + contactRadius && p.y >= b.y0 - contactRadius && p.y <= b.y1 + contactRadius
    }

    /// Centre lines of every painted line, for overlays.
    static func lineSegments() -> [(Pt, Pt)] {
        var segs: [(Pt, Pt)] = []
        let xs = xCentres.map { $0.1 }, ys = yCentres.map { $0.1 }
        for y in ys { segs.append((Pt(xs.first!, y), Pt(xs.last!, y))) }
        for x in [xs[0], xs[1], xs[3], xs[4]] { segs.append((Pt(x, ys.first!), Pt(x, ys.last!))) }
        segs.append((Pt(xs[2], ys[0]), Pt(xs[2], ys[2])))
        segs.append((Pt(xs[2], ys[3]), Pt(xs[2], ys[5])))
        return segs
    }
}

enum Mode: String, Codable, CaseIterable { case singles, doubles
    var cn: String { self == .singles ? "单打" : "双打" }
}

enum Edge: String, Codable { case left, right, near, far
    var cn: String { ["left": "左边线", "right": "右边线", "near": "近端底线", "far": "远端底线"][rawValue]! }
}

struct Pt: Codable, Equatable {
    var x: Double, y: Double
    init(_ x: Double, _ y: Double) { self.x = x; self.y = y }
    static func - (a: Pt, b: Pt) -> Pt { Pt(a.x - b.x, a.y - b.y) }
    static func + (a: Pt, b: Pt) -> Pt { Pt(a.x + b.x, a.y + b.y) }
    static func * (a: Pt, k: Double) -> Pt { Pt(a.x * k, a.y * k) }
    var norm: Double { (x * x + y * y).squareRoot() }
}
