// Runs CourtDetector on still images and writes an overlay PNG next to each.
//   hawkeye-court IMAGE.png [CALIBRATION.json]   (calibration: compare with hand-clicked points)
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

func load(_ path: String) -> (CGImage, [UInt8], Int, Int)? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    let w = img.width, h = img.height
    var buf = [UInt8](repeating: 0, count: w * h * 4)
    let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    return (img, buf, w, h)
}

for path in CommandLine.arguments.dropFirst() where path.hasSuffix(".png") {
    guard let (img, buf, w, h) = load(path) else { print("cannot read \(path)"); continue }
    let t0 = Date()
    let r = buf.withUnsafeBufferPointer { CourtDetector.detect(pixels: $0.baseAddress!, width: w, height: h, rowBytes: w * 4) }
    let dt = Date().timeIntervalSince(t0)
    guard let r else { print("\(path): no court found (\(String(format: "%.2f", dt)) s)"); continue }
    var line = String(format: "%@: %d points, %.0f%% of %d visible model samples on lines, %.2f s", (path as NSString).lastPathComponent, r.points.count, r.score * 100, r.visibleSamples, dt)
    let calPath = (path as NSString).deletingPathExtension + ".cal.json"
    if let d = try? Data(contentsOf: URL(fileURLWithPath: calPath)), let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
       let cp = j["court_points"] as? [String: [Double]] {
        let errs = cp.compactMap { (n, v) -> Double? in Court.keypointByName[n].map { (r.courtToImage.project($0) - Pt(v[0], v[1])).norm } }
        line += String(format: "; vs %d hand-clicked points: median %.1f px, max %.1f px", errs.count, errs.sorted()[errs.count / 2], errs.max() ?? 0)
    }
    print(line)
    // Overlay.
    let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    ctx.translateBy(x: 0, y: CGFloat(h)); ctx.scaleBy(x: 1, y: -1)
    ctx.setStrokeColor(CGColor(red: 1, green: 0, blue: 1, alpha: 1)); ctx.setLineWidth(2)
    for (a, b) in Court.lineSegments() {
        var started = false
        for i in 0...40 {
            let (q, wq) = r.courtToImage.apply(a + (b - a) * (Double(i) / 40))
            let ok = (wq >= 0) == (r.courtToImage.apply(Pt(Court.width / 2, Court.length / 2)).w >= 0)
            if !ok { started = false; continue }
            if started { ctx.addLine(to: CGPoint(x: q.x, y: q.y)) } else { ctx.move(to: CGPoint(x: q.x, y: q.y)); started = true }
        }
        ctx.strokePath()
    }
    ctx.setFillColor(CGColor(red: 0, green: 1, blue: 0, alpha: 1))
    for (_, p) in r.points { ctx.fill(CGRect(x: p.x - 4, y: p.y - 4, width: 8, height: 8)) }
    let out = (path as NSString).deletingPathExtension + ".court.png"
    let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: out) as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, ctx.makeImage()!, nil); CGImageDestinationFinalize(dest)
}
