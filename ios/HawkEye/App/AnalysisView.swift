import SwiftUI

/// Runs TrackNet (Core ML) over the whole video, then landing detection and the
/// in/out calls, and stores the result in the session.
struct AnalysisView: View {
    @EnvironmentObject var store: Store
    @Environment(\.dismiss) private var dismiss
    let id: String
    @State private var progress = 0.0
    @State private var status = "准备中…"
    @State private var running = false
    @State private var cancelled = false
    @State private var error: String?
    @State private var started = Date()

    var body: some View {
        VStack(spacing: 16) {
            ProgressView(value: progress) { Text(status) } currentValueLabel: {
                Text("\(Int(progress * 100))%  ·  已用 \(Int(Date().timeIntervalSince(started))) 秒")
            }
            .padding(.horizontal, 60)
            if let error { Text(error).foregroundStyle(.red) }
            if running { Button("取消") { cancelled = true } }
            Text("分析时请保持 App 在前台。1 分钟 60 fps 的视频大约需要 1 分钟以内。").font(.caption).foregroundStyle(.secondary)
        }
        .navigationTitle("分析")
        .onAppear { if !running { start() } }
        .onDisappear { cancelled = true }
    }

    private func start() {
        guard let s = store.sessions.first(where: { $0.id == id }) else { return }
        guard let cal = Calibration(points: s.calibrationPoints) else { error = "标定无效，请重新标定"; return }
        guard let modelURL = Bundle.main.url(forResource: "TrackNetV3", withExtension: "mlmodelc") else { error = "App 里缺少 TrackNetV3 模型"; return }
        running = true; started = Date(); status = "找球（TrackNet）…"
        let video = store.videoURL(s), mode = s.mode
        let cancelFlag = CancelFlag()
        Task.detached(priority: .userInitiated) {
            do {
                let tracker = try ShuttleTracker(modelURL: modelURL)
                let track = try tracker.track(video: video, progress: { p in
                    Task { @MainActor in progress = p; if cancelled { cancelFlag.value = true } }
                }, isCancelled: { cancelFlag.value })
                if cancelFlag.value { await MainActor.run { running = false; status = "已取消" }; return }
                var settings = Analyzer.Settings(); settings.mode = mode
                let analysis = Analyzer.run(track: track, calibration: cal, settings: settings)
                await MainActor.run {
                    guard var cur = store.sessions.first(where: { $0.id == id }) else { return }
                    cur.analysis = analysis
                    store.save(cur)
                    running = false
                    status = "完成：\(analysis.landings.count) 个落点"
                    dismiss()
                }
            } catch {
                await MainActor.run { running = false; self.error = error.localizedDescription; status = "失败" }
            }
        }
    }
}

final class CancelFlag: @unchecked Sendable { var value = false }
