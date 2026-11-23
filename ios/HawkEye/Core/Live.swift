import CoreML
import Foundation

/// Real-time TrackNetV3. Feed 512x288 BGRA frames in capture order; detections for
/// whole 8-frame batches come back (so results lag by up to 8 frames plus inference).
///
/// The model needs a median background frame. At the start it collects samples for
/// about bootstrapSeconds while holding the frames, then processes the held frames,
/// so nothing at the start of a match is lost. Afterwards the background is
/// refreshed every refreshSeconds from samples of the last ~30 s.
final class LiveTracker {
    static let W = ShuttleTracker.W, H = ShuttleTracker.H, seq = ShuttleTracker.seq
    private let model: MLModel
    private let scaleX: Double, scaleY: Double
    private let plane = W * H
    private let input: MLMultiArray
    private let inp: UnsafeMutablePointer<Float>
    private var slots: [Int] = []
    private var reservoir: [[UInt8]] = []
    private var held: [(Int, [UInt8])] = []
    private(set) var backgroundReady = false
    private let fps: Double
    private var framesSeen = 0
    private var lastRefresh = 0
    var bootstrapSeconds = 4.0, refreshSeconds = 30.0

    init(model: MLModel, fullWidth: Double, fullHeight: Double, fps: Double) throws {
        self.model = model
        self.fps = fps
        scaleX = fullWidth / Double(Self.W)
        scaleY = fullHeight / Double(Self.H)
        input = try MLMultiArray(shape: [1, NSNumber(value: (Self.seq + 1) * 3), NSNumber(value: Self.H), NSNumber(value: Self.W)], dataType: .float32)
        inp = input.dataPointer.assumingMemoryBound(to: Float.self)
    }

    /// Push one frame (BGRA, 512x288, tightly packed). Returns detections that became
    /// available: (frame index, position in full-frame pixels or nil, heatmap max).
    func push(index: Int, bgra: UnsafePointer<UInt8>) throws -> [(Int, Pt?, Float)] {
        framesSeen += 1
        let bootstrapEvery = max(1, Int((fps / 8).rounded()))
        let refreshEvery = max(1, Int((fps / 2).rounded()))
        if !backgroundReady {
            let copy = Array(UnsafeBufferPointer(start: bgra, count: plane * 4))
            held.append((index, copy))
            if framesSeen % bootstrapEvery == 0 { reservoir.append(copy) }
            guard Double(framesSeen) >= bootstrapSeconds * fps, reservoir.count >= 8 else { return [] }
            setMedian()
            backgroundReady = true
            lastRefresh = framesSeen
            var out: [(Int, Pt?, Float)] = []
            let h = held
            held.removeAll()
            for (i, f) in h { out += try f.withUnsafeBufferPointer { try enqueue(index: i, bgra: $0.baseAddress!) } }
            return out
        }
        if framesSeen % refreshEvery == 0 {
            reservoir.append(Array(UnsafeBufferPointer(start: bgra, count: plane * 4)))
            if reservoir.count > 64 { reservoir.removeFirst(reservoir.count - 64) }
        }
        if Double(framesSeen - lastRefresh) >= refreshSeconds * fps && slots.isEmpty {
            setMedian()          // only between batches, so a batch never mixes backgrounds
            lastRefresh = framesSeen
        }
        return try enqueue(index: index, bgra: bgra)
    }

    /// Run the last partial batch (end of a match), padding with the last frame.
    func flush() throws -> [(Int, Pt?, Float)] {
        guard !slots.isEmpty else { return [] }
        let real = slots.count
        let last = (real) * 3 * plane
        for s in real..<Self.seq { (inp + (s + 1) * 3 * plane).update(from: inp + last, count: 3 * plane) }
        return try run(real: real)
    }

    private func enqueue(index: Int, bgra: UnsafePointer<UInt8>) throws -> [(Int, Pt?, Float)] {
        let base = (slots.count + 1) * 3 * plane
        for p in 0..<plane {
            inp[base + p] = Float(bgra[p * 4 + 2]) / 255
            inp[base + plane + p] = Float(bgra[p * 4 + 1]) / 255
            inp[base + 2 * plane + p] = Float(bgra[p * 4]) / 255
        }
        slots.append(index)
        return slots.count == Self.seq ? try run(real: Self.seq) : []
    }

    private func run(real: Int) throws -> [(Int, Pt?, Float)] {
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["frames": input]))
        guard let heat = out.featureValue(for: "heatmaps")?.multiArrayValue else { throw ShuttleTracker.TrackError.modelOutput }
        var res: [(Int, Pt?, Float)] = []
        for s in 0..<real {
            let (pk, c) = ShuttleTracker.peak(heat, slot: s)
            res.append((slots[s], pk.map { Pt($0.x * scaleX, $0.y * scaleY) }, c))
        }
        slots.removeAll()
        return res
    }

    private func setMedian() {
        let n = reservoir.count
        var col = [UInt8](repeating: 0, count: n)
        for p in 0..<plane {
            for (c, bgra) in [(0, 2), (1, 1), (2, 0)] {
                for s in 0..<n { col[s] = reservoir[s][p * 4 + bgra] }
                col.sort()
                let m = n % 2 == 1 ? Float(col[n / 2]) : (Float(col[n / 2 - 1]) + Float(col[n / 2])) / 2
                inp[c * plane + p] = m.rounded(.down) / 255
            }
        }
    }
}

/// Finds confirmed rally landings while the match is running: re-runs the landing
/// detector on the last few seconds of track and reports each landing once, after
/// its rest has been seen (about 1 s after contact).
final class LandingMonitor {
    private(set) var xy: [Pt?] = []
    private(set) var times: [Double] = []
    private var reported: [Int] = []
    let fps: Double, width: Double
    var windowSeconds = 8.0

    init(fps: Double, width: Double) { self.fps = fps; self.width = width }

    /// Detections may arrive in batches; indices must be consecutive from 0.
    func record(index: Int, point: Pt?, time: Double) {
        while xy.count < index { xy.append(nil); times.append(times.last ?? 0) }
        if index == xy.count { xy.append(point); times.append(time) } else if index < xy.count { xy[index] = point; times[index] = time }
    }

    func setTime(index: Int, time: Double) {
        if index < times.count { times[index] = time }
    }

    func poll() -> [LandingCandidate] {
        let n = xy.count
        let lo = max(0, n - Int(windowSeconds * fps))
        guard n - lo > Int(fps) else { return [] }
        var out: [LandingCandidate] = []
        for var c in LandingDetector.detect(Array(xy[lo..<n]), fps: fps, width: width) where c.isRallyLanding {
            c.contactFrame += lo; c.lastFlightFrame += lo; c.restStart += lo; c.restEnd += lo
            if reported.contains(where: { abs($0 - c.contactFrame) <= Int(fps) }) { continue }
            reported.append(c.contactFrame)
            c.contactTime = c.contactFrame < times.count ? times[c.contactFrame] : Double(c.contactFrame) / fps
            out.append(c)
        }
        return out
    }
}
