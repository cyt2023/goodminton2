// Plays a video through the LIVE pipeline (LiveTracker + LandingMonitor) frame by
// frame, as the camera would, and prints each landing when it is confirmed.
//   hawkeye-live VIDEO CALIBRATION.json MODEL.mlpackage [singles|doubles]
import Accelerate
import AVFoundation
import CoreML
import Foundation

let a = CommandLine.arguments
guard a.count >= 4 else { print("usage: hawkeye-live VIDEO CALIBRATION.json MODEL.mlpackage [singles|doubles]"); exit(2) }
let calJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: a[2]))) as! [String: Any]
let cal = Calibration(points: (calJSON["court_points"] as! [String: [Double]]).mapValues { Pt($0[0], $0[1]) })!
let mode = Mode(rawValue: a.count > 4 ? a[4] : "singles") ?? .singles
let compiled = try MLModel.compileModel(at: URL(fileURLWithPath: a[3]))
let cfg = MLModelConfiguration(); cfg.computeUnits = .all
let model = try MLModel(contentsOf: compiled, configuration: cfg)

let asset = AVURLAsset(url: URL(fileURLWithPath: a[1]))
let track = asset.tracks(withMediaType: .video)[0]
let fps = Double(track.nominalFrameRate), size = track.naturalSize
let live = try LiveTracker(model: model, fullWidth: Double(size.width), fullHeight: Double(size.height), fps: fps)
let monitor = LandingMonitor(fps: fps, width: Double(size.width))
let reader = try AVAssetReader(asset: asset)
let out = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
reader.add(out); reader.startReading()
let W = LiveTracker.W, H = LiveTracker.H
let small = UnsafeMutablePointer<UInt8>.allocate(capacity: W * H * 4)
var dst = vImage_Buffer(data: small, height: vImagePixelCount(H), width: vImagePixelCount(W), rowBytes: W * 4)
var i = 0, detected = 0
let t0 = Date()
var settings = Analyzer.Settings(); settings.mode = mode

func handle(_ dets: [(Int, Pt?, Float)]) {
    for (idx, p, _) in dets { monitor.record(index: idx, point: p, time: monitor.times.count > idx ? monitor.times[idx] : Double(idx) / fps); if p != nil { detected += 1 } }
    for c in monitor.poll() {
        let r = Uncertainty.locate(cal, pixel: c.contactPx, cov: Uncertainty.pixelCov(detSigma: settings.detSigma, velocity: c.velocityPxPerFrame),
                                   mode: mode, contactRadius: settings.contactRadius)
        let lag = Double(i - c.contactFrame) / fps
        print(String(format: "LANDING frame %d (%.2f s) confirmed at frame %d (%.2f s later): %@ P(in)=%.3f court (%.2f, %.2f)%@",
                     c.contactFrame, c.contactTime, i, lag, r.call.rawValue, r.pIn, r.courtXY.x, r.courtXY.y, cal.onGround(c.contactPx) ? "" : " [above horizon: would be dropped]"))
    }
}
while let sb = out.copyNextSampleBuffer() {
    guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    var src = vImage_Buffer(data: CVPixelBufferGetBaseAddress(pb), height: vImagePixelCount(CVPixelBufferGetHeight(pb)), width: vImagePixelCount(CVPixelBufferGetWidth(pb)), rowBytes: CVPixelBufferGetBytesPerRow(pb))
    vImageScale_ARGB8888(&src, &dst, nil, vImage_Flags(kvImageHighQualityResampling))
    CVPixelBufferUnlockBaseAddress(pb, .readOnly)
    monitor.record(index: i, point: nil, time: CMSampleBufferGetPresentationTimeStamp(sb).seconds)
    handle(try live.push(index: i, bgra: small))
    i += 1
}
handle(try live.flush())
let dt = Date().timeIntervalSince(t0)
print(String(format: "%d frames in %.1f s (%.0f fps, real time needs %.0f); shuttle detected in %d frames", i, dt, Double(i) / dt, fps, detected))
