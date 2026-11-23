import Accelerate
import AVFoundation
import CoreImage
import CoreML
import SwiftUI
import UIKit

/// Recording quality. 4K gives sharper zoomed snapshots and more precise calibration;
/// 4K 120 needs an iPhone Pro (16 Pro or newer) and fills storage twice as fast.
enum CaptureQuality: String, CaseIterable, Identifiable {
    case uhd60, uhd120, hd60
    var id: String { rawValue }
    var width: Int32 { self == .hd60 ? 1920 : 3840 }
    var height: Int32 { self == .hd60 ? 1080 : 2160 }
    var fps: Double { self == .uhd120 ? 120 : 60 }
    var label: String { ["uhd60": "4K 60", "uhd120": "4K 120", "hd60": "1080p 60"][rawValue]! }
}

extension AVCaptureDevice {
    /// Focus and expose once at a point (0...1 in the sensor's landscape coordinates).
    func hawkFocus(at p: CGPoint) {
        guard (try? lockForConfiguration()) != nil else { return }
        if isFocusPointOfInterestSupported { focusPointOfInterest = p; focusMode = .autoFocus }
        if isExposurePointOfInterestSupported { exposurePointOfInterest = p; exposureMode = .autoExpose }
        unlockForConfiguration()
    }

    /// Lock focus, exposure and white balance so the picture does not change during play.
    func hawkLock(_ on: Bool) {
        guard (try? lockForConfiguration()) != nil else { return }
        if on {
            if isFocusModeSupported(.locked) { focusMode = .locked }
            if isExposureModeSupported(.locked) { exposureMode = .locked }
            if isWhiteBalanceModeSupported(.locked) { whiteBalanceMode = .locked }
        } else {
            if isFocusModeSupported(.continuousAutoFocus) { focusMode = .continuousAutoFocus }
            if isExposureModeSupported(.continuousAutoExposure) { exposureMode = .continuousAutoExposure }
            if isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) { whiteBalanceMode = .continuousAutoWhiteBalance }
        }
        unlockForConfiguration()
    }

    /// Shutter 1/1000 s with ISO raised to keep the brightness. Returns a status line.
    func hawkFastShutter(_ on: Bool) -> String {
        guard (try? lockForConfiguration()) != nil else { return "" }
        defer { unlockForConfiguration() }
        if on, isExposureModeSupported(.custom) {
            let fmt = activeFormat
            let dur = CMTimeMaximum(CMTime(value: 1, timescale: 1000), fmt.minExposureDuration)
            let iso = min(max(self.iso * Float(exposureDuration.seconds / dur.seconds), fmt.minISO), fmt.maxISO)
            setExposureModeCustom(duration: dur, iso: iso, completionHandler: nil)
            return iso >= fmt.maxISO - 1 ? "快门 1/1000 · ISO 已到上限，画面会偏暗" : String(format: "快门 1/1000 · ISO %.0f", iso)
        }
        if isExposureModeSupported(.continuousAutoExposure) { exposureMode = .continuousAutoExposure }
        return "自动曝光"
    }

    func hawkFormat(_ q: CaptureQuality) -> AVCaptureDevice.Format? {
        formats.first { f in
            let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
            return d.width == q.width && d.height == q.height && f.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= q.fps }
        }
    }

    /// Back wide camera at the given resolution and frame rate, auto focus/exposure.
    func hawkConfigure(_ q: CaptureQuality) -> Bool {
        let fmt = hawkFormat(q)
        let fps = q.fps
        guard (try? lockForConfiguration()) != nil else { return false }
        if let fmt { activeFormat = fmt }
        let dur = CMTime(value: 1, timescale: CMTimeScale(fps))
        activeVideoMinFrameDuration = dur
        activeVideoMaxFrameDuration = dur
        if isFocusModeSupported(.continuousAutoFocus) { focusMode = .continuousAutoFocus }
        if isExposureModeSupported(.continuousAutoExposure) { exposureMode = .continuousAutoExposure }
        unlockForConfiguration()
        return fmt != nil
    }
}

/// A live match: camera -> (recording, 2 s ring buffer, TrackNet) -> landing monitor
/// -> in/out call with uncertainty, a snapshot of the contact frame, saved as a session.
///
/// Threads: frames arrive on captureQueue (downscaling, writing, ring buffer); the model
/// and landing logic run on procQueue; published state is set on the main queue.
final class MatchEngine: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    enum Phase { case setup, ready, playing, finishing, finished }
    @Published var phase: Phase = .setup
    @Published var status = ""
    @Published var landings: [Landing] = []
    @Published var banner: Landing?
    @Published var shuttle: Pt?            // latest detection, frame pixels
    @Published var warming = false
    @Published var behindFrames = 0
    @Published var elapsed: Double = 0
    @Published var sessionID: String?
    @Published var locked = false
    @Published var fastShutter = false

    var fps: Double = 60
    var mode: Mode = .singles
    var points: [String: Pt] = [:]
    private(set) var calibration: Calibration?
    /// Two-phone mode: the link to the other phone (nil or role .solo = one phone).
    var link: DualLink?
    var role: DuoRole { link?.role ?? .solo }
    private var fusion: LandingFusion?
    private var fuseTimer: Timer?
    private var startAbs = 0.0          // host-clock time of the first recorded frame (lock)
    private(set) var frameSize = CGSize(width: 1920, height: 1080)

    let session = AVCaptureSession()
    private var device: AVCaptureDevice?
    private let captureQueue = DispatchQueue(label: "match.capture")
    private let procQueue = DispatchQueue(label: "match.process", qos: .userInitiated)
    private let lock = NSLock()
    private let ciContext = CIContext()
    /// No colour management: matches the BGRA frames TrackNet was trained and tested on
    /// (checked on a recorded video: 0.8 levels mean difference).
    private let scaler = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])

    /// Scale a frame to w x h BGRA bytes on the GPU.
    private func render(_ img: CIImage, into buf: inout [UInt8], _ w: Int, _ h: Int, highQuality: Bool) {
        let sx = Double(w) / img.extent.width, sy = Double(h) / img.extent.height
        let scaled = highQuality
            ? img.applyingFilter("CILanczosScaleTransform", parameters: [kCIInputScaleKey: sy, kCIInputAspectRatioKey: sx / sy])
            : img.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        buf.withUnsafeMutableBytes { p in
            scaler.render(scaled, toBitmap: p.baseAddress!, rowBytes: w * 4, bounds: CGRect(x: 0, y: 0, width: w, height: h), format: .BGRA8, colorSpace: nil)
        }
    }
    // Shared between queues, guarded by lock.
    private var latest: CVPixelBuffer?
    private var ring: [(index: Int, bgra: [UInt8])] = []
    private var pending = 0
    // captureQueue only.
    private var playing = false
    private var writer: AVAssetWriter?
    private var writerInput: AVAssetWriterInput?
    private var videoURL: URL?
    private var firstPTS: CMTime?
    private var frameIndex = 0
    // procQueue only.
    private var tracker: LiveTracker?
    private var monitor: LandingMonitor?
    private var detected = 0, processed = 0
    private var folder: URL?
    // main only.
    private var record: SessionRecord?
    private weak var store: Store?
    private var timer: Timer?
    private var started = Date()

    static let ringW = 768, ringH = 432

    // MARK: camera

    /// Qualities this phone's back camera supports (filled in by configure).
    @Published var supportedQualities: [CaptureQuality] = [.uhd60, .hd60]

    func configure(_ quality: CaptureQuality, store: Store) {
        self.fps = quality.fps
        self.store = store
        #if DEBUG
        if let path = ProcessInfo.processInfo.environment["HAWKEYE_LIVE_FILE"] { startFileSource(path); return }
        #endif
        captureQueue.async { [self] in
            session.beginConfiguration()
            session.sessionPreset = .inputPriority
            session.inputs.forEach { session.removeInput($0) }
            guard let cam = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: cam), session.canAddInput(input) else {
                session.commitConfiguration(); publish("没有可用的相机"); return
            }
            session.addInput(input)
            if session.outputs.isEmpty {
                let out = AVCaptureVideoDataOutput()
                // Native YUV: no full-size colour conversion per frame; scaling happens on the GPU.
                out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
                out.alwaysDiscardsLateVideoFrames = true
                out.setSampleBufferDelegate(self, queue: captureQueue)
                if session.canAddOutput(out) { session.addOutput(out) }
            }
            let ok = cam.hawkConfigure(quality)
            let supported = CaptureQuality.allCases.filter { cam.hawkFormat($0) != nil }
            DispatchQueue.main.async { self.supportedQualities = supported }
            for out in session.outputs {
                if let c = out.connection(with: .video) {
                    if c.isVideoStabilizationSupported { c.preferredVideoStabilizationMode = .off }
                    if c.isVideoRotationAngleSupported(0) { c.videoRotationAngle = 0 }
                }
            }
            session.commitConfiguration()
            device = cam
            if !session.isRunning { session.startRunning() }
            publish(ok ? "\(quality.label) · 防抖已关" : "这台手机不支持 \(quality.label)，用默认格式")
        }
    }

    func focus(at p: CGPoint) { captureQueue.async { [self] in device?.hawkFocus(at: p) } ; locked = false; fastShutter = false }
    func setLocked(_ on: Bool) { locked = on; captureQueue.async { [self] in device?.hawkLock(on) } }
    func setFastShutter(_ on: Bool) {
        fastShutter = on
        captureQueue.async { [self] in if let s = device?.hawkFastShutter(on) { publish(s) } }
    }
    func shutdown() { captureQueue.async { [self] in if session.isRunning { session.stopRunning() } } }

    private func publish(_ s: String) { DispatchQueue.main.async { self.status = s } }

    /// Current camera frame, for calibration.
    func snapshot() -> UIImage? {
        lock.lock(); let pb = latest; lock.unlock()
        guard let pb, let cg = ciContext.createCGImage(CIImage(cvPixelBuffer: pb), from: CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))) else { return nil }
        frameSize = CGSize(width: cg.width, height: cg.height)
        return UIImage(cgImage: cg, scale: 1, orientation: .up)
    }

    // MARK: match

    @discardableResult
    func setCalibration(_ pts: [String: Pt]) -> Bool {
        points = pts
        calibration = Calibration(points: pts)
        if calibration != nil && phase == .setup { phase = .ready }
        link?.send(DuoMessage(kind: .status, calibrated: calibration != nil, rms: calibration?.residualRMS))
        return calibration != nil
    }

    @MainActor func start() throws {
        guard let store, calibration != nil else { return }
        lock.lock()
        if let pb = latest { frameSize = CGSize(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb)) }
        lock.unlock()
        guard let url = Bundle.main.url(forResource: "TrackNetV3", withExtension: "mlmodelc") else { throw ShuttleTracker.TrackError.modelOutput }
        let cfg = MLModelConfiguration(); cfg.computeUnits = .all
        let model = try MLModel(contentsOf: url, configuration: cfg)
        let fmt = DateFormatter(); fmt.dateFormat = "yyyyMMdd-HHmmss"
        var rec = SessionRecord(id: fmt.string(from: Date()), created: Date(), videoFile: "video.mov")
        rec.mode = mode
        rec.calibrationPoints = points
        rec.calibrationFrameTime = 0
        rec.note = "实时比赛"
        let dir = store.folder(rec)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        store.save(rec)
        record = rec
        sessionID = rec.id
        landings = []
        let size = frameSize, fps = self.fps
        let tr = try LiveTracker(model: model, fullWidth: Double(size.width), fullHeight: Double(size.height), fps: fps)
        procQueue.sync {
            tracker = tr
            monitor = LandingMonitor(fps: fps, width: Double(size.width))
            folder = dir
            detected = 0; processed = 0
        }
        let video = store.videoURL(rec)
        captureQueue.sync {
            videoURL = video; writer = nil; writerInput = nil; firstPTS = nil; frameIndex = 0
            lock.lock(); ring.removeAll(); pending = 0; lock.unlock()
            playing = true
        }
        phase = .playing
        warming = true
        started = Date()
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.elapsed = Date().timeIntervalSince(self.started)
            self.lock.lock(); self.behindFrames = self.pending; self.lock.unlock()
        }
        if role == .host {
            let st = Analyzer.Settings()
            fusion = LandingFusion(mode: mode, contactRadius: st.contactRadius, alpha: st.alpha)
            fuseTimer = Timer.scheduledTimer(withTimeInterval: 0.3, repeats: true) { [weak self] _ in
                guard let self, let f = self.fusion else { return }
                f.flush(now: DualLink.now()).forEach { self.appendFinal($0) }
            }
            link?.send(DuoMessage(kind: .start, mode: mode))
        }
        #if DEBUG
        fileSourcePlay?()
        #endif
    }

    /// Messages from the other phone (main queue).
    func receive(_ m: DuoMessage) {
        switch m.kind {
        case .start where role == .guest && phase == .ready:
            if let md = m.mode { mode = md }
            MainActor.assumeIsolated { try? start() }
        case .stop where role == .guest:
            stop()
        case .landing where role == .host:
            guard var l = m.landing, let t = m.time, let f = fusion else { return }
            var file: String?
            if let data = m.snapshot, let dir = procQueue.sync(execute: { folder }) {
                let name = "landing_B_\(l.candidate.contactFrame).jpg"
                if (try? data.write(to: dir.appendingPathComponent(name))) != nil { file = name }
            }
            l.snapshotFile = file
            var zfile: String?
            if let data = m.zoom, let dir = procQueue.sync(execute: { folder }) {
                let name = "landing_B_\(l.candidate.contactFrame)_zoom.jpg"
                if (try? data.write(to: dir.appendingPathComponent(name))) != nil { zfile = name }
            }
            l.zoomFile = zfile
            // Put B's landing on this phone's video timeline (its frame numbers are B's own).
            lock.lock(); let t0 = startAbs; lock.unlock()
            l.candidate.contactTime = t - t0
            l.candidate.contactFrame = Int((l.candidate.contactTime * fps).rounded())
            f.add(.init(landing: l, time: t, side: "B", snapshotFile: file, arrived: DualLink.now(), zoomFile: zfile)).forEach { appendFinal($0) }
        default: break
        }
    }

    /// A landing that is final (one phone, or both combined): show and save (main queue).
    private func appendFinal(_ l: Landing) {
        if let i = landings.firstIndex(where: { $0.id == l.id }) { landings[i] = l } else { landings.append(l) }
        banner = l
        UINotificationFeedbackGenerator().notificationOccurred(l.result.call == .ABSTAIN ? .warning : .success)
        MainActor.assumeIsolated { persist() }
    }

    func stop() {
        guard phase == .playing else { return }
        phase = .finishing
        timer?.invalidate()
        fuseTimer?.invalidate()
        if role == .host { link?.send(DuoMessage(kind: .stop)) }
        let group = DispatchGroup()
        group.enter()
        captureQueue.async { [self] in
            playing = false
            if let w = writer, w.status == .writing {
                writerInput?.markAsFinished()
                w.finishWriting { group.leave() }
            } else { group.leave() }
        }
        group.enter()
        procQueue.async { [self] in
            if let t = tracker { handle((try? t.flush()) ?? []) }
            group.leave()
        }
        group.notify(queue: .main) { [self] in
            // Wait a moment for the other phone's last landings, then emit what is left.
            DispatchQueue.main.asyncAfter(deadline: .now() + (role == .host ? 1.5 : 0)) { [self] in
                fusion?.flush(now: DualLink.now(), all: true).forEach { appendFinal($0) }
                MainActor.assumeIsolated { persist() }
                phase = .finished
            }
        }
    }

    // MARK: frames

    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer, from connection: AVCaptureConnection) { ingest(sb) }

    /// One camera frame (captureQueue).
    func ingest(_ sb: CMSampleBuffer) {
        guard let pb = CMSampleBufferGetImageBuffer(sb) else { return }
        lock.lock(); latest = pb; lock.unlock()
        guard playing else { return }
        let pts = CMSampleBufferGetPresentationTimeStamp(sb)
        if writer == nil, let url = videoURL {
            try? FileManager.default.removeItem(at: url)
            if let w = try? AVAssetWriter(outputURL: url, fileType: .mov) {
                let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
                    AVVideoCodecKey: AVVideoCodecType.hevc,
                    AVVideoWidthKey: CVPixelBufferGetWidth(pb), AVVideoHeightKey: CVPixelBufferGetHeight(pb)])
                input.expectsMediaDataInRealTime = true
                if w.canAdd(input) { w.add(input) }
                if w.startWriting() { w.startSession(atSourceTime: pts); writer = w; writerInput = input }
            }
            firstPTS = pts
            lock.lock(); startAbs = pts.seconds; lock.unlock()
        }
        if let wi = writerInput, wi.isReadyForMoreMediaData { wi.append(sb) }
        let t = (pts - (firstPTS ?? pts)).seconds
        let idx = frameIndex
        frameIndex += 1

        let W = LiveTracker.W, H = LiveTracker.H
        var small = [UInt8](repeating: 0, count: W * H * 4)
        var ringBuf = [UInt8](repeating: 0, count: Self.ringW * Self.ringH * 4)
        let src = CIImage(cvPixelBuffer: pb)
        render(src, into: &small, W, H, highQuality: true)
        render(src, into: &ringBuf, Self.ringW, Self.ringH, highQuality: false)

        lock.lock()
        ring.append((idx, ringBuf))
        let keep = Int(2.5 * fps)
        if ring.count > keep { ring.removeFirst(ring.count - keep) }
        // If the model falls behind by more than half a second, skip frames (they count as "not seen").
        let drop = pending > Int(fps / 2)
        if !drop { pending += 1 }
        lock.unlock()
        procQueue.async { [self] in process(index: idx, time: t, small: drop ? nil : small) }
    }

    private func process(index: Int, time: Double, small: [UInt8]?) {
        guard let tracker, let monitor else { return }
        monitor.record(index: index, point: nil, time: time)
        guard let small else { return }
        defer { lock.lock(); pending -= 1; lock.unlock() }
        let dets = (try? small.withUnsafeBufferPointer { try tracker.push(index: index, bgra: $0.baseAddress!) }) ?? []
        processed += 1
        handle(dets)
        let ready = tracker.backgroundReady
        DispatchQueue.main.async { if self.warming != !ready { self.warming = !ready } }
    }

    private func handle(_ dets: [(Int, Pt?, Float)]) {
        guard let monitor, !dets.isEmpty else { return }
        var last: Pt?
        for (i, p, _) in dets {
            monitor.record(index: i, point: p, time: i < monitor.times.count ? monitor.times[i] : Double(i) / fps)
            if let p { detected += 1; last = p }
        }
        let shown = last
        DispatchQueue.main.async { self.shuttle = shown }
        for c in monitor.poll() { landed(c) }
    }

    private func landed(_ c: LandingCandidate) {
        guard let cal = calibration, cal.onGround(c.contactPx),
              Court.signedDistance(cal.toCourt(c.contactPx), .doubles).d > -3 else { return }
        var s = Analyzer.Settings(); s.mode = mode
        let det = s.detSigma(width: Double(frameSize.width))
        let r = Uncertainty.locate(cal, pixel: c.contactPx, cov: Uncertainty.pixelCov(detSigma: det, velocity: c.velocityPxPerFrame),
                                   mode: mode, contactRadius: s.contactRadius, alpha: s.alpha)
        var l = Landing(candidate: c, result: r, pixelSigma: det, playerCall: nil, notALanding: nil)
        lock.lock()
        let frame = ring.min { abs($0.index - c.contactFrame) < abs($1.index - c.contactFrame) }
        lock.unlock()
        if let frame, let folder, let img = renderSnapshot(frame.bgra, cal: cal, landing: l) {
            let name = "landing_\(c.contactFrame).jpg"
            if let data = img.jpegData(compressionQuality: 0.85), (try? data.write(to: folder.appendingPathComponent(name))) != nil {
                l.snapshotFile = name
            }
        }
        // Full-resolution crop of the newest frame: the shuttle usually still lies there.
        lock.lock(); let latestPB = latest; let abs = startAbs + c.contactTime; lock.unlock()
        if let pb = latestPB, let folder, let img = renderZoom(pb, cal: cal, landing: l) {
            let name = "landing_\(c.contactFrame)_zoom.jpg"
            if let data = img.jpegData(compressionQuality: 0.85), (try? data.write(to: folder.appendingPathComponent(name))) != nil { l.zoomFile = name }
        }
        let snapData = l.snapshotFile.flatMap { f in folder.flatMap { try? Data(contentsOf: $0.appendingPathComponent(f)) } }
        let zoomData = l.zoomFile.flatMap { f in folder.flatMap { try? Data(contentsOf: $0.appendingPathComponent(f)) } }
        DispatchQueue.main.async { [self] in
            switch role {
            case .solo:
                appendFinal(l)
            case .guest:
                appendFinal(l)                               // kept on this phone too
                var sent = l
                sent.result = Fusion.rotate(l.result, mode: mode)   // into the host's court coordinates
                #if DEBUG
                if ProcessInfo.processInfo.environment["HAWKEYE_DUO_NOROTATE"] != nil { sent.result = l.result }
                #endif
                sent.snapshotFile = nil
                sent.zoomFile = nil
                link?.send(DuoMessage(kind: .landing, landing: sent, time: link?.toHostClock(abs), snapshot: snapData, zoom: zoomData))
            case .host:
                fusion?.add(.init(landing: l, time: abs, side: "A", snapshotFile: l.snapshotFile, arrived: DualLink.now(), zoomFile: l.zoomFile)).forEach { appendFinal($0) }
            }
        }
    }

    /// Full-resolution crop (a quarter of the frame width) around the landing, with the
    /// boundary, the landing point and its 95% ellipse drawn on it.
    private func renderZoom(_ pb: CVPixelBuffer, cal: Calibration, landing l: Landing) -> UIImage? {
        let W = Double(CVPixelBufferGetWidth(pb)), H = Double(CVPixelBufferGetHeight(pb))
        let cw = (W * 0.25).rounded(), ch = (cw * 9 / 16).rounded()
        let c = l.candidate.contactPx
        let x0 = min(max(0, c.x - cw / 2), W - cw), y0 = min(max(0, c.y - ch / 2), H - ch)    // image coords, top-left
        let ciRect = CGRect(x: x0, y: H - y0 - ch, width: cw, height: ch)                    // Core Image: origin bottom-left
        guard let cg = ciContext.createCGImage(CIImage(cvPixelBuffer: pb), from: ciRect) else { return nil }
        func v(_ p: Pt) -> CGPoint { CGPoint(x: p.x - x0, y: p.y - y0) }
        let fmt = UIGraphicsImageRendererFormat(); fmt.scale = 1
        return UIGraphicsImageRenderer(size: CGSize(width: cw, height: ch), format: fmt).image { ctx in
            UIImage(cgImage: cg).draw(at: .zero)
            let g = ctx.cgContext
            g.setLineWidth(2)
            g.setStrokeColor(UIColor.yellow.withAlphaComponent(0.8).cgColor)
            let b = Court.boundary(mode)
            let corners = [Pt(b.x0, b.y0), Pt(b.x1, b.y0), Pt(b.x1, b.y1), Pt(b.x0, b.y1), Pt(b.x0, b.y0)]
            for i in 0..<4 {
                var started = false
                for k in 0...200 {
                    guard let px = cal.toImage(corners[i] + (corners[i + 1] - corners[i]) * (Double(k) / 200)) else { started = false; continue }
                    if started { g.addLine(to: v(px)) } else { g.move(to: v(px)); started = true }
                }
                g.strokePath()
            }
            let col: UIColor = l.result.call == .IN ? .green : (l.result.call == .OUT ? .red : .orange)
            g.setStrokeColor(col.cgColor); g.setLineWidth(3)
            var started = false
            for q in Uncertainty.ellipse(centre: l.result.courtXY, cov: l.result.cov, steps: 96) {
                guard let px = cal.toImage(q) else { started = false; continue }
                if started { g.addLine(to: v(px)) } else { g.move(to: v(px)); started = true }
            }
            g.strokePath()
            let p = v(c)
            g.move(to: CGPoint(x: p.x - 18, y: p.y)); g.addLine(to: CGPoint(x: p.x + 18, y: p.y))
            g.move(to: CGPoint(x: p.x, y: p.y - 18)); g.addLine(to: CGPoint(x: p.x, y: p.y + 18))
            g.strokePath()
            let label = String(format: " 原画面放大（落地约 1 秒后）· %@ ", l.result.call.cn) as NSString
            label.draw(at: CGPoint(x: 10, y: 10), withAttributes: [.font: UIFont.boldSystemFont(ofSize: max(18, cw / 40)), .foregroundColor: UIColor.white, .backgroundColor: col.withAlphaComponent(0.8)])
        }
    }

    /// Contact frame (768x432) with the boundary, the landing point and its 95% ellipse.
    private func renderSnapshot(_ bgra: [UInt8], cal: Calibration, landing l: Landing) -> UIImage? {
        let w = Self.ringW, h = Self.ringH
        guard let provider = CGDataProvider(data: Data(bgra) as CFData),
              let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else { return nil }
        let sx = Double(w) / Double(frameSize.width), sy = Double(h) / Double(frameSize.height)
        func v(_ p: Pt) -> CGPoint { CGPoint(x: p.x * sx, y: p.y * sy) }
        let fmt = UIGraphicsImageRendererFormat(); fmt.scale = 1
        return UIGraphicsImageRenderer(size: CGSize(width: w, height: h), format: fmt).image { ctx in
            UIImage(cgImage: cg).draw(at: .zero)
            let g = ctx.cgContext
            g.setLineWidth(1.5)
            g.setStrokeColor(UIColor.yellow.cgColor)
            let b = Court.boundary(mode)
            let corners = [Pt(b.x0, b.y0), Pt(b.x1, b.y0), Pt(b.x1, b.y1), Pt(b.x0, b.y1), Pt(b.x0, b.y0)]
            for i in 0..<4 {
                var started = false
                for k in 0...60 {
                    guard let px = cal.toImage(corners[i] + (corners[i + 1] - corners[i]) * (Double(k) / 60)) else { started = false; continue }
                    if started { g.addLine(to: v(px)) } else { g.move(to: v(px)); started = true }
                }
                g.strokePath()
            }
            let col: UIColor = l.result.call == .IN ? .green : (l.result.call == .OUT ? .red : .orange)
            g.setStrokeColor(col.cgColor); g.setLineWidth(2)
            var started = false
            for q in Uncertainty.ellipse(centre: l.result.courtXY, cov: l.result.cov) {
                guard let px = cal.toImage(q) else { started = false; continue }
                if started { g.addLine(to: v(px)) } else { g.move(to: v(px)); started = true }
            }
            g.strokePath()
            let c = v(l.candidate.contactPx)
            g.move(to: CGPoint(x: c.x - 10, y: c.y)); g.addLine(to: CGPoint(x: c.x + 10, y: c.y))
            g.move(to: CGPoint(x: c.x, y: c.y - 10)); g.addLine(to: CGPoint(x: c.x, y: c.y + 10))
            g.strokePath()
            let label = String(format: " %@  P(界内)=%.2f ", l.result.call.cn, l.result.pIn) as NSString
            label.draw(at: CGPoint(x: 10, y: 10), withAttributes: [.font: UIFont.boldSystemFont(ofSize: 22), .foregroundColor: UIColor.white, .backgroundColor: col.withAlphaComponent(0.8)])
        }
    }

    /// Save the session with the landings so far (main queue).
    @MainActor private func persist() {
        guard let store, var rec = record else { return }
        let (det, tot) = procQueue.sync { (detected, processed) }
        rec.analysis = Analysis(landings: landings, nonRally: [], rejected: 0, fps: fps,
                                videoSize: [Double(frameSize.width), Double(frameSize.height)],
                                detectedFrames: det, totalFrames: tot, calibrationRMS: calibration?.residualRMS ?? 0)
        if let i = store.sessions.firstIndex(where: { $0.id == rec.id }) {
            // keep any edits made on the review screens
            let saved = store.sessions[i].analysis?.landings ?? []
            // Keep what was edited on the review screens; everything else is the latest result.
            let merged = landings.map { l -> Landing in
                var n = l
                if let o = saved.first(where: { $0.id == l.id }) { n.playerCall = o.playerCall ?? l.playerCall; n.notALanding = o.notALanding ?? l.notALanding }
                return n
            }
            rec.analysis?.landings = merged
        }
        record = rec
        store.save(rec)
    }

    // MARK: debug video source (simulator tests: HAWKEYE_LIVE_FILE=/path/video.mp4)

    #if DEBUG
    private var fileSourcePlay: (() -> Void)?

    /// Shows the first frame as the "camera", and streams the whole file in real time
    /// through ingest() once the match starts.
    private func startFileSource(_ path: String) {
        let url = URL(fileURLWithPath: path)
        func reader() -> (AVAssetReader, AVAssetReaderTrackOutput)? {
            let asset = AVURLAsset(url: url)
            guard let track = asset.tracks(withMediaType: .video).first, let r = try? AVAssetReader(asset: asset) else { return nil }
            let o = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange])
            r.add(o); r.startReading()
            return (r, o)
        }
        if let fr = Double(exactly: AVURLAsset(url: url).tracks(withMediaType: .video).first?.nominalFrameRate ?? 30) { fps = fr.rounded() }
        captureQueue.async { [self] in
            if let (_, o) = reader(), let sb = o.copyNextSampleBuffer() { ingest(sb) }
            publish("调试：用视频文件代替相机")
        }
        fileSourcePlay = { [weak self] in
            guard let self else { return }
            self.captureQueue.async {
                guard let (_, o) = reader() else { return }
                let t0 = Date()
                var first: Double?
                while self.playing, let sb = o.copyNextSampleBuffer() {
                    let t = CMSampleBufferGetPresentationTimeStamp(sb).seconds
                    if first == nil { first = t }
                    if ProcessInfo.processInfo.environment["HAWKEYE_LIVE_PACED"] != nil {
                        // Slow machines (simulator CPU): wait for the model instead of dropping frames.
                        while true { self.lock.lock(); let p = self.pending; self.lock.unlock(); if p < 4 { break }; Thread.sleep(forTimeInterval: 0.005) }
                    } else {
                        let wait = (t - first!) - Date().timeIntervalSince(t0)
                        if wait > 0 { Thread.sleep(forTimeInterval: wait) }
                    }
                    self.ingest(sb)
                }
            }
        }
    }
    #endif
}
