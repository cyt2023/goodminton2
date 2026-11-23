import AVFoundation
import SwiftUI
import UIKit

extension Call {
    var color: Color { self == .IN ? .green : (self == .OUT ? .red : .orange) }
}

struct ReviewView: View {
    @EnvironmentObject var store: Store
    let id: String
    @State private var showNonRally = false

    var body: some View {
        if let s = store.sessions.first(where: { $0.id == id }), let a = s.analysis {
            List {
                Section {
                    Text(String(format: "%@ · 标定误差 %.1f px · 球在 %d / %d 帧里被找到 · %d 个静止点不在地面或远离球场，已排除",
                                s.mode.cn, a.calibrationRMS, a.detectedFrames, a.totalFrames, a.rejected))
                        .font(.caption).foregroundStyle(.secondary)
                    CourtMap(landings: a.landings, mode: s.mode).frame(height: 220)
                }
                Section("落点") {
                    if a.landings.isEmpty { Text("没有找到落点").foregroundStyle(.secondary) }
                    ForEach(Array(a.landings.enumerated()), id: \.element.id) { i, l in
                        NavigationLink(value: Route.landing(id, l.id, i + 1)) { LandingRow(index: i + 1, l: l) }
                    }
                }
                Section {
                    DisclosureGroup("不像回合落点的候选（\(a.nonRally.count)）", isExpanded: $showNonRally) {
                        ForEach(Array(a.nonRally.enumerated()), id: \.element.id) { i, l in
                            NavigationLink(value: Route.landing(id, l.id, a.landings.count + i + 1)) {
                                VStack(alignment: .leading) {
                                    LandingRow(index: a.landings.count + i + 1, l: l)
                                    Text(l.candidate.reasons.joined(separator: "；")).font(.caption2).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                } footer: { Text("颠球、丢球、回合之间捡球等，不计入落点；有争议时可以在这里回看。") }
            }
            .navigationTitle("回看")
        } else { Text("还没有分析结果") }
    }
}

struct LandingRow: View {
    let index: Int, l: Landing
    var body: some View {
        HStack {
            Text("#\(index)").font(.headline.monospacedDigit()).frame(width: 40, alignment: .leading)
            Text(l.result.call.cn).bold().foregroundStyle(l.result.call.color).frame(width: 56, alignment: .leading)
            VStack(alignment: .leading) {
                Text(String(format: "%.2f s · P(界内) %.2f", l.candidate.contactTime, l.result.pIn))
                Text(String(format: "离%@ %+.0f ± %.0f mm", l.result.nearestEdge.cn, l.result.signedDistance * 1000, l.result.distanceSigma * 1000))
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if let c = l.playerCall { Text("球员：\(["in": "内", "out": "出", "unsure": "不知道"][c] ?? c)").font(.caption) }
            if l.notALanding == true { Text("误报").font(.caption).foregroundStyle(.secondary) }
        }
    }
}

/// Top-down court with the landings and their 95% ellipses.
struct CourtMap: View {
    let landings: [Landing]
    let mode: Mode
    var highlight: Int? = nil
    var body: some View {
        Canvas { ctx, size in
            let margin = 0.8
            let k = min(size.width / (Court.length + 2 * margin), size.height / (Court.width + 2 * margin))
            // Court drawn horizontally: court y -> screen x (near end on the left), court x -> screen y.
            func p(_ q: Pt) -> CGPoint { CGPoint(x: (q.y + margin) * k, y: (Court.width - q.x + margin) * k) }
            ctx.fill(Path(CGRect(x: 0, y: 0, width: (Court.length + 2 * margin) * k, height: (Court.width + 2 * margin) * k)), with: .color(Color(red: 0.2, green: 0.45, blue: 0.3)))
            for (a, b) in Court.lineSegments() {
                var path = Path(); path.move(to: p(a)); path.addLine(to: p(b))
                ctx.stroke(path, with: .color(.white), lineWidth: max(1, Court.line * k))
            }
            let bd = Court.boundary(mode)
            var rect = Path()
            rect.move(to: p(Pt(bd.x0, bd.y0))); [Pt(bd.x1, bd.y0), Pt(bd.x1, bd.y1), Pt(bd.x0, bd.y1), Pt(bd.x0, bd.y0)].forEach { rect.addLine(to: p($0)) }
            ctx.stroke(rect, with: .color(.yellow), lineWidth: 1)
            var net = Path(); net.move(to: p(Pt(-0.3, Court.netY))); net.addLine(to: p(Pt(Court.width + 0.3, Court.netY)))
            ctx.stroke(net, with: .color(.gray), style: StrokeStyle(lineWidth: 2, dash: [4, 3]))
            for (i, l) in landings.enumerated() {
                let col = l.result.call.color
                let e = Uncertainty.ellipse(centre: l.result.courtXY, cov: l.result.cov)
                var ep = Path(); ep.move(to: p(e[0])); e.dropFirst().forEach { ep.addLine(to: p($0)) }
                ctx.stroke(ep, with: .color(col), lineWidth: highlight == l.id ? 2.5 : 1)
                let c = p(l.result.courtXY)
                ctx.fill(Path(ellipseIn: CGRect(x: c.x - 3, y: c.y - 3, width: 6, height: 6)), with: .color(col))
                ctx.draw(Text("\(i + 1)").font(.caption2).foregroundStyle(.white), at: CGPoint(x: c.x + 8, y: c.y - 8))
            }
            ctx.draw(Text("近端").font(.caption2).foregroundStyle(.white), at: CGPoint(x: 14, y: 8))
        }
    }
}

/// Video layer that reports where the picture is drawn, so the overlay can follow it.
struct PlayerView: UIViewRepresentable {
    let player: AVPlayer
    var overlay: (CGRect) -> [OverlayShape]

    struct OverlayShape { var points: [CGPoint]; var color: UIColor; var width: CGFloat; var closed: Bool }

    final class V: UIView {
        override class var layerClass: AnyClass { AVPlayerLayer.self }
        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
        let shapes = CALayer()
        var overlay: ((CGRect) -> [OverlayShape])?
        override func layoutSubviews() {
            super.layoutSubviews()
            shapes.frame = bounds
            redraw()
        }
        func redraw() {
            shapes.sublayers?.forEach { $0.removeFromSuperlayer() }
            guard let overlay else { return }
            for s in overlay(playerLayer.videoRect) where s.points.count > 1 {
                let path = UIBezierPath(); path.move(to: s.points[0]); s.points.dropFirst().forEach { path.addLine(to: $0) }
                if s.closed { path.close() }
                let l = CAShapeLayer(); l.path = path.cgPath; l.strokeColor = s.color.cgColor; l.fillColor = nil; l.lineWidth = s.width
                shapes.addSublayer(l)
            }
        }
    }

    func makeUIView(context: Context) -> V {
        let v = V()
        v.playerLayer.player = player
        v.playerLayer.videoGravity = .resizeAspect
        v.layer.addSublayer(v.shapes)
        v.overlay = overlay
        return v
    }
    func updateUIView(_ v: V, context: Context) { v.overlay = overlay; v.setNeedsLayout() }
}

struct LandingDetailView: View {
    @EnvironmentObject var store: Store
    let id: String, frame: Int, index: Int
    @State private var player = AVQueuePlayer()
    @State private var looper: AVPlayerLooper?
    @State private var slow = true

    private var session: SessionRecord? { store.sessions.first { $0.id == id } }
    private var landing: Landing? {
        guard let a = session?.analysis else { return nil }
        return (a.landings + a.nonRally).first { $0.id == frame }
    }

    var body: some View {
        if let s = session, let l = landing, let a = s.analysis, let cal = Calibration(points: s.calibrationPoints) {
            HStack(spacing: 0) {
                PlayerView(player: player) { rect in overlayShapes(rect: rect, video: a.videoSize, cal: cal, l: l, mode: s.mode) }
                    .background(Color.black)
                ScrollView {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("#\(index) \(l.result.call.cn)").font(.title2.bold()).foregroundStyle(l.result.call.color)
                        Text(String(format: "P(界内) = %.3f", l.result.pIn))
                        Text(String(format: "离%@ %+.0f ± %.0f mm（正 = 界内）", l.result.nearestEdge.cn, l.result.signedDistance * 1000, l.result.distanceSigma * 1000))
                        Text(String(format: "触地 %.2f s（第 %d 帧）", l.candidate.contactTime, l.candidate.contactFrame)).font(.caption)
                        if let r = l.result.abstainReason { Text("判不了：\(r)").font(.caption).foregroundStyle(.orange) }
                        if !l.candidate.reasons.isEmpty { Text("不像回合落点：" + l.candidate.reasons.joined(separator: "；")).font(.caption).foregroundStyle(.secondary) }
                        HStack {
                            Button(slow ? "1/4 速" : "原速") { slow.toggle(); player.rate = slow ? 0.25 : 1 }
                            Button("定格触地帧") {
                                player.pause()
                                player.seek(to: CMTime(seconds: l.candidate.contactTime, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
                            }
                            Button("播放") { player.rate = slow ? 0.25 : 1 }
                        }.buttonStyle(.bordered)
                        CourtMap(landings: [l], mode: s.mode, highlight: l.id).frame(height: 150)
                        Divider()
                        Text("记录（研究用）").font(.headline)
                        Text("球员当场怎么判的：").font(.caption)
                        HStack {
                            ForEach([("in", "内"), ("out", "出"), ("unsure", "不知道")], id: \.0) { key, label in
                                Button(label) { update { $0.playerCall = $0.playerCall == key ? nil : key } }
                                    .buttonStyle(.bordered).tint(l.playerCall == key ? .yellow : .gray)
                            }
                        }
                        Toggle("这不是落点（误报）", isOn: Binding(get: { l.notALanding ?? false }, set: { v in update { $0.notALanding = v } }))
                    }
                    .padding()
                }
                .frame(width: 300)
            }
            .navigationTitle("落点 #\(index)")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear { start(url: store.videoURL(s), t: l.candidate.contactTime) }
            .onDisappear { player.pause() }
        } else { Text("找不到这个落点") }
    }

    private func start(url: URL, t: Double) {
        let item = AVPlayerItem(asset: AVURLAsset(url: url))
        let range = CMTimeRange(start: CMTime(seconds: max(0, t - 1.0), preferredTimescale: 600),
                                end: CMTime(seconds: t + 0.6, preferredTimescale: 600))
        looper = AVPlayerLooper(player: player, templateItem: item, timeRange: range)
        player.isMuted = true
        player.play()
        player.rate = slow ? 0.25 : 1
    }

    private func update(_ f: (inout Landing) -> Void) {
        guard var s = session, var a = s.analysis else { return }
        if let i = a.landings.firstIndex(where: { $0.id == frame }) { f(&a.landings[i]) }
        else if let i = a.nonRally.firstIndex(where: { $0.id == frame }) { f(&a.nonRally[i]) }
        s.analysis = a
        store.save(s)
    }

    /// Court boundary, the landing point and its 95% ellipse, in view coordinates.
    private func overlayShapes(rect: CGRect, video: [Double], cal: Calibration, l: Landing, mode: Mode) -> [PlayerView.OverlayShape] {
        guard rect.width > 0, video.count == 2 else { return [] }
        let sx = rect.width / video[0], sy = rect.height / video[1]
        func v(_ px: Pt) -> CGPoint { CGPoint(x: rect.minX + px.x * sx, y: rect.minY + px.y * sy) }
        func polyline(_ court: [Pt]) -> [[CGPoint]] {   // split where a point has no image (beyond the horizon)
            var runs: [[CGPoint]] = [], cur: [CGPoint] = []
            for q in court { if let px = cal.toImage(q) { cur.append(v(px)) } else if !cur.isEmpty { runs.append(cur); cur = [] } }
            if !cur.isEmpty { runs.append(cur) }
            return runs
        }
        var out: [PlayerView.OverlayShape] = []
        let b = Court.boundary(mode)
        let corners = [Pt(b.x0, b.y0), Pt(b.x1, b.y0), Pt(b.x1, b.y1), Pt(b.x0, b.y1), Pt(b.x0, b.y0)]
        for i in 0..<4 {
            let edge = (0...60).map { corners[i] + (corners[i + 1] - corners[i]) * (Double($0) / 60) }
            for run in polyline(edge) { out.append(.init(points: run, color: .yellow, width: 1, closed: false)) }
        }
        let col: UIColor = l.result.call == .IN ? .green : (l.result.call == .OUT ? .red : .orange)
        for run in polyline(Uncertainty.ellipse(centre: l.result.courtXY, cov: l.result.cov)) {
            out.append(.init(points: run, color: col, width: 2, closed: false))
        }
        let c = v(l.candidate.contactPx), r: CGFloat = 8
        out.append(.init(points: [CGPoint(x: c.x - r, y: c.y), CGPoint(x: c.x + r, y: c.y)], color: col, width: 2, closed: false))
        out.append(.init(points: [CGPoint(x: c.x, y: c.y - r), CGPoint(x: c.x, y: c.y + r)], color: col, width: 2, closed: false))
        return out
    }
}
