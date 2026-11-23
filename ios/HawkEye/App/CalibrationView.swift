import AVFoundation
import SwiftUI
import UIKit

/// Frame of a video in display orientation, exact time.
func grabFrame(_ url: URL, at t: Double) -> UIImage? {
    let gen = AVAssetImageGenerator(asset: AVURLAsset(url: url))
    gen.appliesPreferredTrackTransform = true
    gen.requestedTimeToleranceBefore = .zero
    gen.requestedTimeToleranceAfter = .zero
    guard let cg = try? gen.copyCGImage(at: CMTime(seconds: t, preferredTimescale: 600), actualTime: nil) else { return nil }
    return UIImage(cgImage: cg, scale: 1, orientation: .up)   // 1 point = 1 video pixel
}

/// Zoomable image; taps reported in image pixels; overlay drawn in image pixels.
struct ZoomableImage: UIViewRepresentable {
    let image: UIImage
    var points: [String: Pt]
    var selected: String?
    var lines: [[Pt]]
    var onTap: (Pt) -> Void

    final class Overlay: UIView {
        var points: [String: Pt] = [:], selected: String?, lines: [[Pt]] = [], zoom: CGFloat = 1
        override func draw(_ rect: CGRect) {
            guard let g = UIGraphicsGetCurrentContext() else { return }
            g.setLineWidth(1.5 / zoom)
            g.setStrokeColor(UIColor.magenta.withAlphaComponent(0.9).cgColor)
            for l in lines where l.count > 1 {
                g.move(to: CGPoint(x: l[0].x, y: l[0].y))
                for p in l.dropFirst() { g.addLine(to: CGPoint(x: p.x, y: p.y)) }
                g.strokePath()
            }
            let r = 10 / zoom
            for (n, p) in points {
                g.setStrokeColor((n == selected ? UIColor.yellow : UIColor.green).cgColor)
                g.setLineWidth(2 / zoom)
                g.move(to: CGPoint(x: p.x - r, y: p.y)); g.addLine(to: CGPoint(x: p.x + r, y: p.y))
                g.move(to: CGPoint(x: p.x, y: p.y - r)); g.addLine(to: CGPoint(x: p.x, y: p.y + r))
                g.strokePath()
            }
        }
    }

    /// Calls back on every layout, so the fit can follow the final size of the view.
    final class FitScrollView: UIScrollView {
        var onLayout: (() -> Void)?
        override func layoutSubviews() { super.layoutSubviews(); onLayout?() }
    }

    final class Coordinator: NSObject, UIScrollViewDelegate {
        var parent: ZoomableImage
        let container = UIView(), imageView = UIImageView(), overlay = Overlay()
        private var fittedFor: CGSize = .zero
        private var userZoomed = false
        func scrollViewWillBeginZooming(_ s: UIScrollView, with view: UIView?) { userZoomed = true }

        /// Whole frame visible and centred; redone when the view size changes
        /// (unless the user has zoomed in since).
        func refit(_ s: UIScrollView) {
            let size = s.bounds.size, img = imageView.image?.size ?? .zero
            guard size.width > 1, size.height > 1, img.width > 0 else { return }
            let fit = min(size.width / img.width, size.height / img.height)
            s.minimumZoomScale = fit
            s.maximumZoomScale = max(fit * 12, 1)
            if size != fittedFor && !userZoomed || s.zoomScale < fit {
                s.zoomScale = fit
                fittedFor = size
            }
            overlay.zoom = s.zoomScale
            center(s)
        }

        /// New image of a different size: reset the content.
        func setImage(_ image: UIImage, in s: UIScrollView) {
            guard imageView.image !== image else { return }
            let sizeChanged = imageView.image?.size != image.size
            imageView.image = image
            if sizeChanged {
                s.zoomScale = 1
                container.frame = CGRect(origin: .zero, size: image.size)
                imageView.frame = container.bounds
                overlay.frame = container.bounds
                s.contentSize = image.size
                fittedFor = .zero
                userZoomed = false
            }
            refit(s)
        }
        init(_ p: ZoomableImage) { parent = p }
        func viewForZooming(in scrollView: UIScrollView) -> UIView? { container }
        func scrollViewDidZoom(_ s: UIScrollView) { overlay.zoom = s.zoomScale; overlay.setNeedsDisplay(); center(s) }
        /// Keep the frame centred when it is smaller than the view.
        func center(_ s: UIScrollView) {
            let dx = max(0, (s.bounds.width - container.frame.width) / 2), dy = max(0, (s.bounds.height - container.frame.height) / 2)
            s.contentInset = UIEdgeInsets(top: dy, left: dx, bottom: dy, right: dx)
        }
        @objc func tapped(_ g: UITapGestureRecognizer) {
            let p = g.location(in: container)
            guard p.x >= 0, p.y >= 0, p.x <= container.bounds.width, p.y <= container.bounds.height else { return }
            parent.onTap(Pt(Double(p.x), Double(p.y)))
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> UIScrollView {
        let sv = FitScrollView()
        sv.delegate = context.coordinator
        sv.maximumZoomScale = 12
        sv.backgroundColor = .black
        let c = context.coordinator
        c.overlay.backgroundColor = .clear
        c.overlay.isUserInteractionEnabled = false
        c.container.addSubview(c.imageView)
        c.container.addSubview(c.overlay)
        sv.addSubview(c.container)
        sv.addGestureRecognizer(UITapGestureRecognizer(target: c, action: #selector(Coordinator.tapped(_:))))
        sv.onLayout = { [weak sv, weak c] in if let sv, let c { c.refit(sv) } }
        c.setImage(image, in: sv)
        return sv
    }

    func updateUIView(_ sv: UIScrollView, context: Context) {
        let c = context.coordinator
        c.parent = self
        c.setImage(image, in: sv)
        c.overlay.points = points; c.overlay.selected = selected; c.overlay.lines = lines
        c.overlay.setNeedsDisplay()
    }
}

/// RGBA bytes of an image (for the court-line detector).
func rgbaPixels(_ image: UIImage) -> (bytes: [UInt8], width: Int, height: Int)? {
    guard let cg = image.cgImage else { return nil }
    let w = cg.width, h = cg.height
    var buf = [UInt8](repeating: 0, count: w * h * 4)
    guard let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                              space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    return (buf, w, h)
}

/// Small court diagram: near end at the bottom, the corner to tap highlighted.
struct CornerGuide: View {
    let step: Int
    var body: some View {
        Canvas { ctx, size in
            let pad = 10.0, cw = size.height * 0.45, ch = size.height - 2 * pad - 14
            let x0 = (size.width - cw) / 2
            let r = CGRect(x: x0, y: pad, width: cw, height: ch)
            ctx.fill(Path(r), with: .color(Color(red: 0.2, green: 0.5, blue: 0.35)))
            ctx.stroke(Path(r), with: .color(.white), lineWidth: 1.5)
            var net = Path(); net.move(to: CGPoint(x: x0 - 4, y: pad + ch / 2)); net.addLine(to: CGPoint(x: x0 + cw + 4, y: pad + ch / 2))
            ctx.stroke(net, with: .color(.white.opacity(0.7)), style: StrokeStyle(lineWidth: 1, dash: [3, 2]))
            let corners = [CGPoint(x: x0, y: pad + ch), CGPoint(x: x0 + cw, y: pad + ch), CGPoint(x: x0 + cw, y: pad), CGPoint(x: x0, y: pad)]
            for (i, c) in corners.enumerated() {
                let rad = i == step ? 7.0 : 4.0
                ctx.fill(Path(ellipseIn: CGRect(x: c.x - rad, y: c.y - rad, width: 2 * rad, height: 2 * rad)), with: .color(i < step ? .green : (i == step ? .red : .gray)))
            }
            ctx.draw(Text("手机这一端").font(.caption2), at: CGPoint(x: size.width / 2, y: size.height - 7))
        }
    }
}

/// Calibration: automatic line detection first; if it is wrong, tap the four outer court
/// corners (the court model is then snapped onto the painted lines); single points can be
/// fixed in the list. Used for recorded videos (CalibrationView) and the live camera.
struct CalibrationEditor<Extra: View>: View {
    let image: UIImage?
    @Binding var points: [String: Pt]
    var saveTitle = "保存"
    @ViewBuilder var extra: () -> Extra
    var onSave: () -> Void
    @State private var selected = Court.keypoints[0].name
    @State private var lines: [[Pt]] = []
    @State private var message = ""
    @State private var prep: CourtDetector.Prepared?
    @State private var preparedFor: UIImage?
    @State private var cornerStep: Int?
    @State private var busy = false
    @State private var showList = false

    static var corners: [(String, String)] {
        [("back_near/doubles_left", "近端左角"), ("back_near/doubles_right", "近端右角"),
         ("back_far/doubles_right", "远端右角"), ("back_far/doubles_left", "远端左角")]
    }

    var body: some View {
        HStack(spacing: 0) {
            ZStack {
                if let image {
                    ZoomableImage(image: image, points: points, selected: cornerStep == nil ? selected : nil, lines: lines) { p in tapped(p) }
                } else { ProgressView() }
                if busy { ProgressView("识别场地线…").padding().background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10)) }
            }
            VStack(alignment: .leading, spacing: 8) {
                Text(message).font(.caption).fixedSize(horizontal: false, vertical: true)
                extra()
                if let k = cornerStep {
                    HStack(alignment: .top) {
                        CornerGuide(step: k).frame(width: 90, height: 140)
                        VStack(alignment: .leading, spacing: 6) {
                            Text("点\(Self.corners[k].1)").font(.headline)
                            Text("双打外边线和底线的交点（外沿也行，放大点更准）。左右按站在手机这一端看过去。").font(.caption2)
                            Button("取消") { cornerStep = nil; message = "" }.font(.caption)
                        }
                    }
                } else {
                    HStack {
                        Button { autoDetect() } label: { Label("自动识别", systemImage: "wand.and.stars") }
                        Button { startCorners() } label: { Label("点四个角", systemImage: "hand.tap") }
                    }
                    .buttonStyle(.bordered).font(.caption)
                    DisclosureGroup("逐个修改交点（\(points.count)）", isExpanded: $showList) {
                        List(Court.keypoints, id: \.name) { kp in
                            Button { selected = kp.name } label: {
                                HStack {
                                    Image(systemName: points[kp.name] != nil ? "checkmark.circle.fill" : "circle")
                                        .foregroundStyle(points[kp.name] != nil ? .green : .secondary)
                                    Text(Court.label(kp.name)).font(.caption)
                                }
                            }
                            .listRowBackground(kp.name == selected ? Color.yellow.opacity(0.3) : Color.clear)
                        }
                        .listStyle(.plain).frame(minHeight: 160)
                    }
                    .font(.caption)
                }
                Spacer(minLength: 0)
                HStack {
                    if showList && cornerStep == nil {
                        Button("删掉这个点") { points[selected] = nil; lines = [] }.disabled(points[selected] == nil)
                    }
                    Spacer()
                    Button("检查") { check() }.disabled(points.count < 4)
                    Button(saveTitle) { if check() != nil { onSave() } }.buttonStyle(.borderedProminent).disabled(points.count < 4 || cornerStep != nil)
                }
                .font(.callout)
            }
            .frame(width: 300)
            .padding(8)
        }
        .onAppear { prepareAndMaybeDetect() }
        .onChange(of: image) { _, _ in prepareAndMaybeDetect() }
    }

    private func tapped(_ p: Pt) {
        if let k = cornerStep {
            points[Self.corners[k].0] = p
            lines = []
            if k < 3 { cornerStep = k + 1 } else { cornerStep = nil; finishCorners() }
        } else {
            guard showList else { message = "要改单个交点：先展开\"逐个修改交点\"，选中一个，再点画面。"; return }
            points[selected] = p
            lines = []
            advance()
        }
    }

    /// Line mask for the current image (background), then auto-detect if nothing is set yet.
    private func prepareAndMaybeDetect() {
        guard let image, preparedFor !== image else { return }
        preparedFor = image
        busy = true
        let img = image
        DispatchQueue.global(qos: .userInitiated).async {
            let p = rgbaPixels(img).map { px in px.bytes.withUnsafeBufferPointer { CourtDetector.prepare(pixels: $0.baseAddress!, width: px.width, height: px.height, rowBytes: px.width * 4) } }
            DispatchQueue.main.async {
                prep = p
                busy = false
                if points.count >= 4 { check() } else { autoDetect() }
            }
        }
    }

    private func autoDetect() {
        guard let image else { return }
        busy = true
        let img = image
        DispatchQueue.global(qos: .userInitiated).async {
            let r = rgbaPixels(img).flatMap { px in px.bytes.withUnsafeBufferPointer { CourtDetector.detect(pixels: $0.baseAddress!, width: px.width, height: px.height, rowBytes: px.width * 4) } }
            DispatchQueue.main.async {
                busy = false
                // In the intended setup the whole court is in view: a fit that leaves most of
                // the model outside the picture, or off the painted lines, is not shown.
                let total = CourtDetector.modelSamples(step: 0.15).count
                if let r, Double(r.visibleSamples) >= 0.6 * Double(total), r.score >= 0.5 {
                    points = r.points
                    check()
                    message = "自动识别了 \(r.points.count) 个交点。请仔细看粉色线是否和这块场地的线完全重合；不对就点\"点四个角\"。"
                } else {
                    startCorners()
                    message = "自动识别不确定（画面里没有找到完整的场地）。请按顺序点场地四个外角，App 会补全其余交点。"
                }
            }
        }
    }

    private func startCorners() {
        points = [:]; lines = []
        cornerStep = 0
        message = "按顺序点场地四个外角，App 会自动补全其余交点。"
    }

    private func finishCorners() {
        let four = points
        if let prep, let r = CourtDetector.refine(points: four, prep) {
            points = r.points
            check()
            message = String(format: "已根据四个角补全 %d 个交点。看粉色线是否重合；个别点不准可以在\"逐个修改交点\"里改。", r.points.count)
        } else {
            check()
            message = "已用四个角标定。看粉色线是否重合。"
        }
    }

    private func advance() {
        if let i = Court.keypoints.firstIndex(where: { $0.name == selected }) {
            let rest = Court.keypoints[(i + 1)...] + Court.keypoints[..<i]
            if let n = rest.first(where: { points[$0.name] == nil }) { selected = n.name }
        }
    }

    /// Fit and draw every court line back onto the frame.
    @discardableResult
    private func check() -> Calibration? {
        guard let cal = Calibration(points: points) else {
            message = "拟合失败：点可能在一条直线上，或者点错了交点。"
            lines = []
            return nil
        }
        lines = Court.lineSegments().map { a, b in
            (0...40).compactMap { i -> Pt? in cal.toImage(a + (b - a) * (Double(i) / 40)) }
        }
        if cal.residualRMS > 5 {
            message = String(format: "%d 个点，平均误差 %.1f px，偏大：多半有点点错了交点。可以重新\"点四个角\"。", cal.nPoints, cal.residualRMS)
        } else if message.isEmpty || message.hasPrefix("拟合失败") || message.contains("平均误差") {
            message = String(format: "%d 个点，平均误差 %.1f px。看粉色线是否和场地线重合；重合就保存。", cal.nPoints, cal.residualRMS)
        }
        return cal
    }
}

struct CalibrationView: View {
    @EnvironmentObject var store: Store
    @Environment(\.dismiss) private var dismiss
    let id: String
    @State private var image: UIImage?
    @State private var t = 1.0
    @State private var duration = 1.0
    @State private var points: [String: Pt] = [:]
    @State private var loaded = false

    private var session: SessionRecord? { store.sessions.first { $0.id == id } }

    var body: some View {
        Group {
            if loaded {
                CalibrationEditor(image: image, points: $points, extra: {
                    HStack {
                        Text("画面时间").font(.caption)
                        Slider(value: $t, in: 0...max(duration, 0.1)) { editing in if !editing { loadFrame() } }
                    }
                }, onSave: save)
            } else { ProgressView() }
        }
        .navigationTitle("标定场地")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            guard let s = session, !loaded else { return }
            points = s.calibrationPoints
            t = s.calibrationFrameTime
            duration = AVURLAsset(url: store.videoURL(s)).duration.seconds
            loaded = true
            loadFrame()
        }
    }

    private func loadFrame() {
        guard let s = session else { return }
        let url = store.videoURL(s), time = t
        DispatchQueue.global(qos: .userInitiated).async {
            let img = grabFrame(url, at: time)
            DispatchQueue.main.async { image = img }
        }
    }

    private func save() {
        guard var s = session else { return }
        s.calibrationPoints = points
        s.calibrationFrameTime = t
        s.analysis = nil
        store.save(s)
        dismiss()
    }
}
