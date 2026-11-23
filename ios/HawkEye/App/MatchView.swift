import AVFoundation
import SwiftUI
import UIKit

/// Camera preview with the court boundary and the live shuttle position drawn on top.
struct LivePreview: UIViewRepresentable {
    let session: AVCaptureSession
    let calibration: Calibration?
    let mode: Mode
    let frameSize: CGSize
    let shuttle: Pt?
    var onTap: (CGPoint) -> Void

    final class V: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var preview: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
        let court = CAShapeLayer(), dot = CAShapeLayer()
        var draw: (() -> Void)?
        var onTap: ((CGPoint) -> Void)?
        override func layoutSubviews() { super.layoutSubviews(); court.frame = bounds; dot.frame = bounds; draw?() }
        @objc func tapped(_ g: UITapGestureRecognizer) { onTap?(preview.captureDevicePointConverted(fromLayerPoint: g.location(in: self))) }
    }

    func makeUIView(context: Context) -> V {
        let v = V()
        v.backgroundColor = .black
        v.preview.session = session
        v.preview.videoGravity = .resizeAspect
        if let c = v.preview.connection, c.isVideoRotationAngleSupported(0) { c.videoRotationAngle = 0 }
        v.court.strokeColor = UIColor.systemYellow.cgColor; v.court.fillColor = nil; v.court.lineWidth = 1.5
        v.dot.fillColor = UIColor.systemOrange.cgColor
        v.layer.addSublayer(v.court); v.layer.addSublayer(v.dot)
        v.addGestureRecognizer(UITapGestureRecognizer(target: v, action: #selector(V.tapped(_:))))
        return v
    }

    func updateUIView(_ v: V, context: Context) {
        v.onTap = onTap
        if let c = v.preview.connection, c.isVideoRotationAngleSupported(0) { c.videoRotationAngle = 0 }
        let cal = calibration, mode = self.mode, size = frameSize, shuttle = self.shuttle
        v.draw = { [weak v] in
            guard let v, size.width > 0 else { return }
            func layerPt(_ px: Pt) -> CGPoint {
                v.preview.layerPointConverted(fromCaptureDevicePoint: CGPoint(x: px.x / size.width, y: px.y / size.height))
            }
            let path = UIBezierPath()
            if let cal {
                let b = Court.boundary(mode)
                let corners = [Pt(b.x0, b.y0), Pt(b.x1, b.y0), Pt(b.x1, b.y1), Pt(b.x0, b.y1), Pt(b.x0, b.y0)]
                for i in 0..<4 {
                    var started = false
                    for k in 0...40 {
                        guard let px = cal.toImage(corners[i] + (corners[i + 1] - corners[i]) * (Double(k) / 40)) else { started = false; continue }
                        if started { path.addLine(to: layerPt(px)) } else { path.move(to: layerPt(px)); started = true }
                    }
                }
            }
            v.court.path = path.cgPath
            if let s = shuttle { let p = layerPt(s); v.dot.path = UIBezierPath(ovalIn: CGRect(x: p.x - 5, y: p.y - 5, width: 10, height: 10)).cgPath }
            else { v.dot.path = nil }
        }
        v.draw?()
    }
}

/// Live match: frame the court -> calibrate on a camera frame -> play; every landing is
/// called as it happens, and the replay button shows the last one (snapshot + 3D).
struct MatchView: View {
    @EnvironmentObject var store: Store
    var onFinish: (String?) -> Void
    @StateObject private var engine = MatchEngine()
    @StateObject private var link = DualLink()
    @State private var role: DuoRole = .solo
    @State private var fps = 60.0
    @State private var mode: Mode = .singles
    @State private var calibrating = false
    @State private var calImage: UIImage?
    @State private var calPoints: [String: Pt] = [:]
    @State private var replayIndex: Int?
    @State private var error: String?

    var body: some View {
        ZStack {
            LivePreview(session: engine.session, calibration: engine.calibration, mode: mode, frameSize: engine.frameSize,
                        shuttle: engine.phase == .playing ? engine.shuttle : nil) { p in
                if engine.phase != .playing { engine.focus(at: p) }
            }
            .ignoresSafeArea()
            HStack(alignment: .top) {
                panel
                    .padding(12).frame(width: 250, alignment: .leading)
                    .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 14))
                    .foregroundStyle(.white).tint(.yellow)
                Spacer()
                if engine.phase == .playing || engine.phase == .finished { rightSide }
            }
            .padding(12)
        }
        .statusBarHidden()
        .fullScreenCover(isPresented: $calibrating) {
            NavigationStack {
                CalibrationEditor(image: calImage, points: $calPoints, saveTitle: "用这个标定", extra: {
                    Picker("比赛", selection: $mode) { ForEach(Mode.allCases, id: \.self) { Text($0.cn).tag($0) } }.pickerStyle(.segmented)
                }, onSave: {
                    engine.mode = mode
                    engine.setCalibration(calPoints)
                    calibrating = false
                })
                .navigationTitle("标定场地（点场地线交点）")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar { ToolbarItem(placement: .cancellationAction) { Button("取消") { calibrating = false } } }
            }
        }
        .fullScreenCover(isPresented: Binding(get: { replayIndex != nil }, set: { if !$0 { replayIndex = nil } })) {
            if let i = replayIndex, !engine.landings.isEmpty {
                ReplayView(folder: engine.sessionID.flatMap { id in store.sessions.first { $0.id == id } }.map { store.folder($0) },
                           landings: engine.landings, mode: engine.mode, index: min(i, engine.landings.count - 1))
            }
        }
        .alert("出错了", isPresented: Binding(get: { error != nil }, set: { if !$0 { error = nil } })) { Button("好") {} } message: { Text(error ?? "") }
        .onAppear(perform: setup)
        .onDisappear { engine.shutdown(); link.stop() }
        .onChange(of: engine.landings.count) { _, n in debugAutoReplay(n) }
    }

    @ViewBuilder private var panel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button { finishAndLeave() } label: { Image(systemName: "xmark.circle.fill").font(.title2) }
                    .disabled(engine.phase == .finishing)
                Text("实时比赛").font(.headline)
            }
            switch engine.phase {
            case .setup, .ready:
                Picker("模式", selection: $role) { ForEach(DuoRole.allCases, id: \.self) { Text($0.cn).tag($0) } }
                    .pickerStyle(.menu)
                    .onChange(of: role) { _, r in link.start(r); engine.link = r == .solo ? nil : link }
                if role != .solo { duoStatus }
                Text(role == .solo ? "1 手机横放在三脚架上，斜对着要判的线，整块场地尽量都在画面里。点画面对焦。"
                                   : "1 两部手机各放在一端底线后面的中间，三脚架尽量高，整块场地都在画面里。点画面对焦。").font(.caption)
                Picker("帧率", selection: $fps) { Text("60 fps").tag(60.0); Text("30 fps").tag(30.0) }
                    .pickerStyle(.segmented)
                    .onChange(of: fps) { _, v in engine.configure(fps: v, store: store) }
                Toggle("锁定曝光/对焦", isOn: Binding(get: { engine.locked }, set: { engine.setLocked($0) }))
                Toggle("快门 1/1000", isOn: Binding(get: { engine.fastShutter }, set: { engine.setFastShutter($0) }))
                Text("2 在画面上点场地线交点（先点几个角）。").font(.caption)
                Button(engine.calibration == nil ? "标定场地" : "重新标定") {
                    calImage = engine.snapshot()
                    calPoints = engine.points
                    calibrating = calImage != nil
                    if calImage == nil { error = "还没有相机画面" }
                }
                .buttonStyle(.bordered)
                if let cal = engine.calibration {
                    Text(String(format: "已标定 %d 点，误差 %.1f px · %@", cal.nPoints, cal.residualRMS, engine.mode.cn)).font(.caption).foregroundStyle(.green)
                    if role == .guest {
                        Text("3 标定好了，等 A 端开始比赛（会自动开始）。").font(.caption).foregroundStyle(.yellow)
                    } else {
                        Text("3 开始比赛后不要再碰手机。前 4 秒在建立背景。").font(.caption)
                        if role == .host && !link.peerCalibrated { Text("B 端还没连上或还没标定，开始后只用这一部手机。").font(.caption2).foregroundStyle(.orange) }
                        Button("开始比赛") { startMatch() }.buttonStyle(.borderedProminent)
                    }
                }
                Text(engine.status).font(.caption2).foregroundStyle(.secondary)
            case .playing:
                Text(String(format: "%02d:%02d", Int(engine.elapsed) / 60, Int(engine.elapsed) % 60)).font(.title.monospacedDigit())
                if engine.warming { Label("建立背景中…", systemImage: "hourglass").font(.caption) }
                Text("落点 \(engine.landings.count) 个 · 界内 \(engine.landings.filter { $0.result.call == .IN }.count) · 出界 \(engine.landings.filter { $0.result.call == .OUT }.count)")
                    .font(.caption)
                if role != .solo { duoStatus }
                if engine.behindFrames > Int(fps / 4) { Text("处理跟不上（落后 \(engine.behindFrames) 帧），可以换 30 fps").font(.caption2).foregroundStyle(.orange) }
                Button("结束比赛") { engine.stop() }.buttonStyle(.bordered).tint(.red)
            case .finishing:
                ProgressView("保存中…")
            case .finished:
                Text("比赛结束：\(engine.landings.count) 个落点，录像已保存。").font(.caption)
                Button("查看全部落点") { onFinish(engine.sessionID) }.buttonStyle(.borderedProminent)
            }
        }
    }

    @ViewBuilder private var duoStatus: some View {
        VStack(alignment: .leading, spacing: 2) {
            if link.connected {
                Label("已连接：\(link.peerName ?? "")", systemImage: "link").font(.caption).foregroundStyle(.green)
                if role == .guest, let rtt = link.roundTrip { Text(String(format: "时钟已对齐（误差 < %.0f ms）", max(1, (rtt * 500).rounded(.up)))).font(.caption2) }
                if role == .host { Text(link.peerCalibrated ? String(format: "B 端已标定（误差 %.1f px）", link.peerRMS ?? 0) : "B 端还没标定").font(.caption2) }
            } else {
                Label(role == .host ? "等待 B 端连接…" : "正在找 A 端…", systemImage: "antenna.radiowaves.left.and.right").font(.caption).foregroundStyle(.orange)
            }
        }
    }

    @ViewBuilder private var rightSide: some View {
        VStack(alignment: .trailing, spacing: 10) {
            if let b = engine.banner {
                VStack(alignment: .trailing, spacing: 2) {
                    Text(b.result.call.cn).font(.system(size: 30, weight: .heavy))
                    Text(String(format: "%.1f s · P(界内) %.2f", b.candidate.contactTime, b.result.pIn)).font(.caption)
                }
                .padding(.horizontal, 14).padding(.vertical, 8)
                .background(b.result.call.color.opacity(0.9), in: RoundedRectangle(cornerRadius: 12))
                .foregroundStyle(.white)
            }
            Button { replayIndex = engine.landings.count - 1 } label: {
                Label("回放", systemImage: "play.rectangle.fill").font(.title3.bold()).padding(.horizontal, 10).padding(.vertical, 6)
            }
            .buttonStyle(.borderedProminent)
            .disabled(engine.landings.isEmpty)
        }
    }

    private func setup() {
        link.onMessage = { m in engine.receive(m) }
        AVCaptureDevice.requestAccess(for: .video) { ok in
            DispatchQueue.main.async {
                if ok || ProcessInfo.processInfo.environment["HAWKEYE_LIVE_FILE"] != nil { engine.configure(fps: fps, store: store) }
                else { error = "没有相机权限：设置 › 隐私 › 相机" }
                debugPreset()
            }
        }
    }

    private func startMatch() {
        do { try engine.start() } catch { self.error = "无法开始：\(error.localizedDescription)" }
    }

    private func finishAndLeave() {
        if engine.phase == .playing {
            engine.stop()
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { onFinish(engine.sessionID) }
        } else {
            onFinish(engine.phase == .finished ? engine.sessionID : nil)
        }
    }

    // Debug builds only: HAWKEYE_LIVE_CAL=<calibration.json> presets the calibration,
    // HAWKEYE_LIVE_AUTOSTART=1 starts the match, HAWKEYE_LIVE_AUTOREPLAY=1 opens the
    // replay at the first landing (simulator tests without tapping).
    private func debugPreset() {
        #if DEBUG
        let env = ProcessInfo.processInfo.environment
        if let f = env["HAWKEYE_LIVE_CAL"], let d = try? Data(contentsOf: URL(fileURLWithPath: f)),
           let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any], let cp = j["court_points"] as? [String: [Double]] {
            if let r = env["HAWKEYE_DUO_ROLE"].flatMap(DuoRole.init(rawValue:)) { role = r; link.start(r); engine.link = link }
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                engine.setCalibration(cp.mapValues { Pt($0[0], $0[1]) })
                guard env["HAWKEYE_LIVE_AUTOSTART"] != nil, role != .guest else { return }
                // Host: wait (up to 30 s) until the other phone is calibrated.
                func tryStart(_ n: Int) {
                    if role != .host || link.peerCalibrated || n > 60 { startMatch() }
                    else { DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { tryStart(n + 1) } }
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { tryStart(0) }
            }
        }
        #endif
    }

    private func debugAutoReplay(_ n: Int) {
        #if DEBUG
        if n == 1, ProcessInfo.processInfo.environment["HAWKEYE_LIVE_AUTOREPLAY"] != nil { replayIndex = 0 }
        #endif
    }
}
