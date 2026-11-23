import AVFoundation
import SwiftUI

/// Camera for court recordings: back wide camera, 1080p at 60 (or 30) fps,
/// video stabilisation OFF (it changes the image geometry), tap to focus/expose,
/// optional lock and fast shutter (less motion blur on the shuttle).
final class CameraController: NSObject, ObservableObject, AVCaptureFileOutputRecordingDelegate {
    let session = AVCaptureSession()
    private let output = AVCaptureMovieFileOutput()
    private var device: AVCaptureDevice?
    private let queue = DispatchQueue(label: "camera")
    @Published var isRecording = false
    @Published var status = ""
    @Published var locked = false
    @Published var fastShutter = false
    @Published var elapsed: TimeInterval = 0
    var onFinished: ((URL?) -> Void)?
    private var timer: Timer?
    private var started = Date()

    func configure(fps: Double) {
        queue.async { [self] in
            session.beginConfiguration()
            session.sessionPreset = .inputPriority
            session.inputs.forEach { session.removeInput($0) }
            guard let cam = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: cam), session.canAddInput(input) else {
                session.commitConfiguration(); publish("没有可用的相机"); return
            }
            session.addInput(input)
            if let mic = AVCaptureDevice.default(for: .audio), let mi = try? AVCaptureDeviceInput(device: mic), session.canAddInput(mi) {
                session.addInput(mi)   // the sound helps align phones and find players' calls later
            }
            if session.outputs.isEmpty, session.canAddOutput(output) { session.addOutput(output) }
            let ok = cam.hawkConfigure(fps: fps)
            if let c = output.connection(with: .video) {
                if c.isVideoStabilizationSupported { c.preferredVideoStabilizationMode = .off }
                if c.isVideoRotationAngleSupported(0) { c.videoRotationAngle = 0 }   // landscape, home indicator on the right
            }
            session.commitConfiguration()
            device = cam
            if !session.isRunning { session.startRunning() }
            publish(!ok ? "这台手机不支持 1080p \(Int(fps)) fps，用默认格式" : "1080p · \(Int(fps)) fps · 防抖已关")
        }
    }

    private func publish(_ s: String) { DispatchQueue.main.async { self.status = s } }

    func focus(at p: CGPoint) {
        queue.async { [self] in device?.hawkFocus(at: p) }
        locked = false; fastShutter = false
    }

    func setLocked(_ on: Bool) { queue.async { [self] in device?.hawkLock(on) } }

    func setFastShutter(_ on: Bool) { queue.async { [self] in if let s = device?.hawkFastShutter(on) { publish(s) } } }

    func toggleRecording() {
        if output.isRecording { output.stopRecording(); return }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("rec-\(UUID().uuidString).mov")
        queue.async { [self] in
            if let c = output.connection(with: .video) {
                if c.isVideoStabilizationSupported { c.preferredVideoStabilizationMode = .off }
                if c.isVideoRotationAngleSupported(0) { c.videoRotationAngle = 0 }
            }
            output.startRecording(to: url, recordingDelegate: self)
        }
    }

    func fileOutput(_ output: AVCaptureFileOutput, didStartRecordingTo fileURL: URL, from connections: [AVCaptureConnection]) {
        DispatchQueue.main.async {
            self.isRecording = true; self.started = Date(); self.elapsed = 0
            self.timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in self.elapsed = Date().timeIntervalSince(self.started) }
        }
    }

    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL, from connections: [AVCaptureConnection], error: Error?) {
        DispatchQueue.main.async {
            self.timer?.invalidate(); self.isRecording = false
            // A file can still be complete when error is set (e.g. max duration reached).
            let ok = error == nil || ((error as NSError?)?.userInfo[AVErrorRecordingSuccessfullyFinishedKey] as? Bool ?? false)
            self.onFinished?(ok ? outputFileURL : nil)
        }
    }

    func stop() { queue.async { [self] in if session.isRunning { session.stopRunning() } } }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    var onTap: (CGPoint) -> Void

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
        var onTap: ((CGPoint) -> Void)?
        @objc func tapped(_ g: UITapGestureRecognizer) {
            onTap?(previewLayer.captureDevicePointConverted(fromLayerPoint: g.location(in: self)))
        }
    }

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.previewLayer.session = session
        v.previewLayer.videoGravity = .resizeAspect
        if let c = v.previewLayer.connection, c.isVideoRotationAngleSupported(0) { c.videoRotationAngle = 0 }
        v.onTap = onTap
        v.addGestureRecognizer(UITapGestureRecognizer(target: v, action: #selector(PreviewView.tapped(_:))))
        return v
    }
    func updateUIView(_ v: PreviewView, context: Context) {
        v.onTap = onTap
        if let c = v.previewLayer.connection, c.isVideoRotationAngleSupported(0) { c.videoRotationAngle = 0 }
    }
}

struct RecorderView: View {
    var onDone: (URL?) -> Void
    @StateObject private var cam = CameraController()
    @State private var fps = 60.0

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            CameraPreview(session: cam.session) { p in cam.focus(at: p) }.ignoresSafeArea()
            HStack {
                VStack(alignment: .leading, spacing: 10) {
                    Button { cam.stop(); onDone(nil) } label: { Image(systemName: "xmark.circle.fill").font(.title) }
                        .disabled(cam.isRecording)
                    Picker("帧率", selection: $fps) { Text("60 fps").tag(60.0); Text("30 fps").tag(30.0) }
                        .pickerStyle(.segmented).frame(width: 150).disabled(cam.isRecording)
                    Toggle("锁定曝光/对焦", isOn: Binding(get: { cam.locked }, set: { cam.locked = $0; cam.setLocked($0) }))
                        .frame(width: 200)
                    Toggle("快门 1/1000", isOn: Binding(get: { cam.fastShutter }, set: { cam.fastShutter = $0; cam.setFastShutter($0) }))
                        .frame(width: 200)
                    Text(cam.status).font(.caption)
                    Text("点画面对焦。先对准要判的线，再锁定。").font(.caption2).foregroundStyle(.secondary)
                    Spacer()
                }
                .padding().foregroundStyle(.white).tint(.yellow)
                .background(LinearGradient(colors: [.black.opacity(0.6), .clear], startPoint: .leading, endPoint: .trailing))
                Spacer()
                VStack {
                    Spacer()
                    if cam.isRecording { Text(String(format: "%02d:%02d", Int(cam.elapsed) / 60, Int(cam.elapsed) % 60)).font(.headline.monospacedDigit()).foregroundStyle(.red) }
                    Button { cam.toggleRecording() } label: {
                        ZStack {
                            Circle().stroke(.white, lineWidth: 4).frame(width: 74, height: 74)
                            if cam.isRecording { RoundedRectangle(cornerRadius: 6).fill(.red).frame(width: 30, height: 30) }
                            else { Circle().fill(.red).frame(width: 60, height: 60) }
                        }
                    }
                    Spacer()
                }.padding(.trailing, 24)
            }
        }
        .onAppear {
            cam.onFinished = { url in cam.stop(); onDone(url) }
            AVCaptureDevice.requestAccess(for: .video) { ok in
                AVCaptureDevice.requestAccess(for: .audio) { _ in
                    DispatchQueue.main.async { if ok { cam.configure(fps: fps) } else { cam.status = "没有相机权限：设置 › 隐私 › 相机" } }
                }
            }
        }
        .onChange(of: fps) { _, v in cam.configure(fps: v) }
        .statusBarHidden()
    }
}
