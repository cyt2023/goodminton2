import Accelerate
import AVFoundation
import CoreML
import CoreVideo
import Foundation

/// TrackNetV3 on device (Core ML), same preprocessing as the author's
/// Video_IterableDataset with bg_mode=concat: RGB frames resized to 512x288,
/// a median background frame first, 8 frames per inference, values / 255.
/// Detection = centre of the largest heatmap blob above 0.5, as in the author's
/// predict_location().
final class ShuttleTracker {
    static let W = 512, H = 288, seq = 8

    struct Track: Codable {
        var fps: Double
        var width: Double, height: Double     // display size (after the video's rotation)
        var times: [Double]                   // presentation time of every frame (s)
        var xy: [Pt?]                         // display pixels, nil = not detected
        var confidence: [Float]
    }

    enum TrackError: LocalizedError {
        case noVideoTrack, portrait, readFailed(String), modelOutput
        var errorDescription: String? {
            switch self {
            case .noVideoTrack: return "视频里没有画面轨道"
            case .portrait: return "这是竖屏视频。请横屏拍摄（TrackNet 只支持横向画面）"
            case .readFailed(let s): return "读视频失败：\(s)"
            case .modelOutput: return "模型输出格式不对"
            }
        }
    }

    let model: MLModel

    init(modelURL: URL) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = ProcessInfo.processInfo.environment["HAWKEYE_CPU_ONLY"] != nil ? .cpuOnly : .all
        var url = modelURL
        if url.pathExtension == "mlpackage" || url.pathExtension == "mlmodel" {
            url = try MLModel.compileModel(at: url)
        }
        model = try MLModel(contentsOf: url, configuration: cfg)
    }

    /// Decode every frame as BGRA, downscaled to 512x288. Calls body(index, time, smallBGRA).
    private func decode(_ asset: AVURLAsset, body: (Int, Double, UnsafeMutableRawPointer) throws -> Bool) throws {
        guard let track = asset.tracks(withMediaType: .video).first else { throw TrackError.noVideoTrack }
        let reader = try AVAssetReader(asset: asset)
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        out.alwaysCopiesSampleData = false
        reader.add(out)
        guard reader.startReading() else { throw TrackError.readFailed(reader.error?.localizedDescription ?? "?") }
        let small = UnsafeMutableRawPointer.allocate(byteCount: Self.W * Self.H * 4, alignment: 64)
        defer { small.deallocate() }
        var dst = vImage_Buffer(data: small, height: vImagePixelCount(Self.H), width: vImagePixelCount(Self.W), rowBytes: Self.W * 4)
        var i = 0
        while let sb = out.copyNextSampleBuffer() {
            guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
            let t = CMSampleBufferGetPresentationTimeStamp(sb).seconds
            CVPixelBufferLockBaseAddress(pb, .readOnly)
            var src = vImage_Buffer(data: CVPixelBufferGetBaseAddress(pb), height: vImagePixelCount(CVPixelBufferGetHeight(pb)),
                                    width: vImagePixelCount(CVPixelBufferGetWidth(pb)), rowBytes: CVPixelBufferGetBytesPerRow(pb))
            let err = vImageScale_ARGB8888(&src, &dst, nil, vImage_Flags(kvImageHighQualityResampling))
            CVPixelBufferUnlockBaseAddress(pb, .readOnly)
            guard err == kvImageNoError else { throw TrackError.readFailed("vImageScale \(err)") }
            if try !body(i, t, small) { reader.cancelReading(); return }
            i += 1
        }
        if reader.status == .failed { throw TrackError.readFailed(reader.error?.localizedDescription ?? "?") }
    }

    func track(video: URL, progress: (Double) -> Void = { _ in }, isCancelled: () -> Bool = { false }) throws -> Track {
        let asset = AVURLAsset(url: video)
        guard let vt = asset.tracks(withMediaType: .video).first else { throw TrackError.noVideoTrack }
        let natural = vt.naturalSize, tf = vt.preferredTransform
        // Display frame = natural frame under the rotation (with or without translation in tf).
        let frameRect = CGRect(origin: .zero, size: natural).applying(tf)
        let dw = frameRect.width, dh = frameRect.height
        if dh > dw { throw TrackError.portrait }
        let fps = Double(vt.nominalFrameRate > 0 ? vt.nominalFrameRate : 30)
        let estCount = max(1, Int(asset.duration.seconds * fps))
        let plane = Self.W * Self.H

        // Pass 1: median background from <= 64 evenly spaced frames.
        let step = max(1, estCount / 64)
        var samples: [[UInt8]] = []
        try decode(asset) { i, _, px in
            if i % step == 0 && samples.count < 64 {
                samples.append(Array(UnsafeBufferPointer(start: px.assumingMemoryBound(to: UInt8.self), count: plane * 4)))
            }
            if i % 30 == 0 { progress(0.1 * Double(i) / Double(estCount)) }
            return !isCancelled()
        }
        var median = [Float](repeating: 0, count: plane * 3)   // RGB planes, /255
        var col = [UInt8](repeating: 0, count: samples.count)
        for p in 0..<plane {
            for (c, bgra) in [(0, 2), (1, 1), (2, 0)] {        // R, G, B from BGRA
                for s in 0..<samples.count { col[s] = samples[s][p * 4 + bgra] }
                col.sort()
                let n = col.count
                let m = n % 2 == 1 ? Float(col[n / 2]) : (Float(col[n / 2 - 1]) + Float(col[n / 2])) / 2
                median[c * plane + p] = m.rounded(.down) / 255      // the author casts the median to uint8
            }
        }
        samples.removeAll()

        // Pass 2: TrackNet on 8-frame batches.
        let input = try MLMultiArray(shape: [1, NSNumber(value: (Self.seq + 1) * 3), NSNumber(value: Self.H), NSNumber(value: Self.W)], dataType: .float32)
        let inp = input.dataPointer.assumingMemoryBound(to: Float.self)
        median.withUnsafeBufferPointer { inp.update(from: $0.baseAddress!, count: plane * 3) }
        var times: [Double] = [], xy: [Pt?] = [], conf: [Float] = []
        var batch = 0
        let sx = Double(natural.width) / Double(Self.W), sy = Double(natural.height) / Double(Self.H)
        var lastFrame = [Float](repeating: 0, count: plane * 3)

        func fill(slot: Int, from px: UnsafeMutableRawPointer) {
            let b = px.assumingMemoryBound(to: UInt8.self)
            let base = (slot + 1) * 3 * plane
            for p in 0..<plane {
                inp[base + p] = Float(b[p * 4 + 2]) / 255
                inp[base + plane + p] = Float(b[p * 4 + 1]) / 255
                inp[base + 2 * plane + p] = Float(b[p * 4]) / 255
            }
        }
        func run(real: Int) throws {
            let outFeat = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["frames": input]))
            guard let heat = outFeat.featureValue(for: "heatmaps")?.multiArrayValue else { throw TrackError.modelOutput }
            if ProcessInfo.processInfo.environment["HAWKEYE_DEBUG"] != nil && times.count <= 64 {
                var s0 = 0.0, s1 = 0.0
                for k in 0..<plane { s0 += Double(inp[k]); s1 += Double(inp[3 * plane + k]) }
                let p0 = Self.peak(heat, slot: 0)
                print("DEBUG batch@\(times.count) median mean \(s0 / Double(plane)) frame0 mean \(s1 / Double(plane)) out \(heat.dataType.rawValue) shape \(heat.shape) strides \(heat.strides) max0 \(p0.1)")
            }
            for s in 0..<real {
                let (pk, c) = Self.peak(heat, slot: s)
                conf.append(c)
                if let pk {
                    // TrackNet works on the stored (natural) frame; map to display coordinates.
                    let q = CGPoint(x: pk.x * sx, y: pk.y * sy).applying(tf)
                    xy.append(Pt(Double(q.x - frameRect.minX), Double(q.y - frameRect.minY)))
                } else { xy.append(nil) }
            }
        }
        try decode(asset) { i, t, px in
            fill(slot: batch, from: px)
            times.append(t)
            batch += 1
            if batch == Self.seq {
                try run(real: Self.seq)
                batch = 0
            }
            if i % 16 == 0 { progress(0.1 + 0.9 * min(1, Double(i) / Double(estCount))) }
            return !isCancelled()
        }
        if batch > 0 {   // pad the last batch by repeating the last real frame (author behaviour)
            let lastBase = batch * 3 * plane
            for k in 0..<(plane * 3) { lastFrame[k] = inp[lastBase + k] }
            for slot in batch..<Self.seq { lastFrame.withUnsafeBufferPointer { (inp + (slot + 1) * 3 * plane).update(from: $0.baseAddress!, count: plane * 3) } }
            try run(real: batch)
        }
        progress(1)
        return Track(fps: fps, width: Double(dw), height: Double(dh), times: times, xy: xy, confidence: conf)
    }

    /// Centre of the largest (bounding-box area) 8-connected blob with value > 0.5, in 512x288 px.
    static func peak(_ heat: MLMultiArray, slot: Int) -> (Pt?, Float) {
        let W = Self.W, H = Self.H
        let strides = heat.strides.map { $0.intValue }
        var vals = [Float](repeating: 0, count: W * H)
        let base = slot * strides[1]
        // withUnsafeBytes, not dataPointer: Core ML may back the output with its own
        // buffer, and dataPointer then does not show the values (all zeros on the simulator).
        heat.withUnsafeBytes { raw in
            if heat.dataType == .float16 {
                let p = raw.bindMemory(to: Float16.self)
                for y in 0..<H { for x in 0..<W { vals[y * W + x] = Float(p[base + y * strides[2] + x * strides[3]]) } }
            } else {
                let p = raw.bindMemory(to: Float.self)
                for y in 0..<H { for x in 0..<W { vals[y * W + x] = p[base + y * strides[2] + x * strides[3]] } }
            }
        }
        let maxV = vals.max() ?? 0
        guard maxV > 0.5 else { return (nil, maxV) }
        var seen = [Bool](repeating: false, count: W * H)
        var best: (area: Int, x0: Int, y0: Int, x1: Int, y1: Int)? = nil
        var stack: [Int] = []
        for start in 0..<(W * H) where vals[start] > 0.5 && !seen[start] {
            var x0 = W, y0 = H, x1 = -1, y1 = -1
            stack.append(start); seen[start] = true
            while let q = stack.popLast() {
                let x = q % W, y = q / W
                x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
                for dy in -1...1 { for dx in -1...1 where dx != 0 || dy != 0 {
                    let nx = x + dx, ny = y + dy
                    guard nx >= 0, ny >= 0, nx < W, ny < H else { continue }
                    let r = ny * W + nx
                    if vals[r] > 0.5 && !seen[r] { seen[r] = true; stack.append(r) }
                } }
            }
            let area = (x1 - x0 + 1) * (y1 - y0 + 1)
            if best == nil || area > best!.area { best = (area, x0, y0, x1, y1) }
        }
        guard let b = best else { return (nil, maxV) }
        // cv2.boundingRect gives w = x1 - x0 + 1; the author's centre is x + w/2 (int-truncated later).
        let cx = Double(b.x0) + Double(b.x1 - b.x0 + 1) / 2, cy = Double(b.y0) + Double(b.y1 - b.y0 + 1) / 2
        return (Pt(cx, cy), maxV)
    }
}
