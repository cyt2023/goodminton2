// Mac harness for the app's Core code: same tracker, landing detector and
// uncertainty as on the iPhone.
//   hawkeye-cli VIDEO CALIBRATION.json MODEL.mlpackage [singles|doubles] [out.json]
// CALIBRATION.json: {"court_points": {"back_near/doubles_left": [x, y], ...}} (as written by calibrate_server.py)
import Foundation

let a = CommandLine.arguments
guard a.count >= 4 else { print("usage: hawkeye-cli VIDEO CALIBRATION.json MODEL.mlpackage [singles|doubles] [out.json]"); exit(2) }
let video = URL(fileURLWithPath: a[1]), calFile = URL(fileURLWithPath: a[2]), modelURL = URL(fileURLWithPath: a[3])
let mode = Mode(rawValue: a.count > 4 ? a[4] : "singles") ?? .singles
let calJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: calFile)) as! [String: Any]
let pts = (calJSON["court_points"] as! [String: [Double]]).mapValues { Pt($0[0], $0[1]) }
guard let cal = Calibration(points: pts) else { print("calibration failed"); exit(1) }
print(String(format: "calibration: %d points, RMS %.2f px, click sigma %.2f px", cal.nPoints, cal.residualRMS, cal.clickSigma))
let t0 = Date()
let tracker = try ShuttleTracker(modelURL: modelURL)
var lastP = -1
let track = try tracker.track(video: video) { p in let q = Int(p * 10); if q != lastP { lastP = q; print("  progress \(q * 10)%") } }
let dt = Date().timeIntervalSince(t0)
print(String(format: "tracked %d frames in %.1f s (%.0f fps); detected in %d", track.xy.count, dt, Double(track.xy.count) / dt, track.xy.compactMap { $0 }.count))
var s = Analyzer.Settings(); s.mode = mode
let an = Analyzer.run(track: track, calibration: cal, settings: s)
for (title, ls) in [("rally landings", an.landings), ("non-rally", an.nonRally)] {
    print("\(title): \(ls.count)")
    for l in ls {
        let r = l.result
        print(String(format: "  frame %d (%.2f s) px (%.0f, %.0f) -> court (%.2f, %.2f) m  %@ P(in)=%.3f  d=%+.0f±%.0f mm to %@ %@",
                     l.candidate.contactFrame, l.candidate.contactTime, l.candidate.contactPx.x, l.candidate.contactPx.y,
                     r.courtXY.x, r.courtXY.y, r.call.rawValue, r.pIn, r.signedDistance * 1000, r.distanceSigma * 1000,
                     r.nearestEdge.rawValue, l.candidate.reasons.joined(separator: "; ")))
    }
}
print("rejected (off floor / far outside): \(an.rejected)")
if a.count > 5 {
    let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    try enc.encode(["track": track]).write(to: URL(fileURLWithPath: a[5]))
}
