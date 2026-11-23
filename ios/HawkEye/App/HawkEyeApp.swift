import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

enum Route: Hashable {
    case session(String), calibrate(String), analyze(String), review(String), landing(String, Int, Int)
}

@main
struct HawkEyeApp: App {
    @StateObject private var store = Store()
    var body: some Scene {
        WindowGroup { ContentView().environmentObject(store) }
    }
}

/// A movie picked from Photos, copied to a temporary file.
struct PickedMovie: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { SentTransferredFile($0.url) } importing: { received in
            let dst = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + "." + (received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension))
            try FileManager.default.copyItem(at: received.file, to: dst)
            return PickedMovie(url: dst)
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var store: Store
    @State private var path: [Route] = []
    @State private var showRecorder = false
    @State private var showMatch = false
    @State private var pickerItem: PhotosPickerItem?
    @State private var showFiles = false
    @State private var busy = false
    @State private var error: String?

    var body: some View {
        NavigationStack(path: $path) {
            List {
                Section {
                    Button { showMatch = true } label: { Label("开始一场比赛（实时判界）", systemImage: "sportscourt.fill").font(.headline) }
                } footer: {
                    Text("标定场地后开始比赛，每次球落地都会实时判界内/出界，可以随时回放上一个落点（截图 + 3D）。")
                }
                Section {
                    Button { showRecorder = true } label: { Label("录一段新视频", systemImage: "video.fill") }
                    PhotosPicker(selection: $pickerItem, matching: .videos) { Label("从相册导入视频", systemImage: "photo.on.rectangle") }
                    Button { showFiles = true } label: { Label("从文件导入视频", systemImage: "folder") }
                } footer: {
                    Text("横屏、三脚架固定、整条要判的线都在画面里。普通视频即可（1080p 60 fps 最好）。")
                }
                Section("记录") {
                    if store.sessions.isEmpty { Text("还没有记录").foregroundStyle(.secondary) }
                    ForEach(store.sessions) { s in
                        NavigationLink(value: Route.session(s.id)) { SessionRow(s: s) }
                    }
                    .onDelete { idx in idx.map { store.sessions[$0] }.forEach(store.delete) }
                }
            }
            .navigationTitle("手机鹰眼")
            .navigationDestination(for: Route.self) { r in
                switch r {
                case .session(let id): SessionView(id: id)
                case .calibrate(let id): CalibrationView(id: id)
                case .analyze(let id): AnalysisView(id: id)
                case .review(let id): ReviewView(id: id)
                case .landing(let id, let frame, let index): LandingDetailView(id: id, frame: frame, index: index)
                }
            }
            .onAppear(perform: openFromEnvironment)
            .overlay { if busy { ProgressView("导入中…").padding().background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12)) } }
            .alert("出错了", isPresented: Binding(get: { error != nil }, set: { if !$0 { error = nil } })) { Button("好") {} } message: { Text(error ?? "") }
        }
        .fullScreenCover(isPresented: $showMatch) {
            MatchView { id in
                showMatch = false
                store.load()
                if let id { path = [.session(id), .review(id)] }
            }
        }
        .fullScreenCover(isPresented: $showRecorder) {
            RecorderView { url in
                showRecorder = false
                guard let url else { return }
                do { let s = try store.create(fromVideo: url, move: true); path = [.session(s.id)] } catch { self.error = error.localizedDescription }
            }
        }
        .onChange(of: pickerItem) { _, item in
            guard let item else { return }
            busy = true
            Task {
                defer { busy = false; pickerItem = nil }
                do {
                    guard let m = try await item.loadTransferable(type: PickedMovie.self) else { throw CocoaError(.fileReadUnknown) }
                    let s = try store.create(fromVideo: m.url, move: true)
                    path = [.session(s.id)]
                } catch { self.error = "导入失败：\(error.localizedDescription)" }
            }
        }
        .fileImporter(isPresented: $showFiles, allowedContentTypes: [.movie, .mpeg4Movie, .quickTimeMovie]) { result in
            switch result {
            case .success(let url):
                let ok = url.startAccessingSecurityScopedResource()
                defer { if ok { url.stopAccessingSecurityScopedResource() } }
                do { let s = try store.create(fromVideo: url, move: false); path = [.session(s.id)] } catch { self.error = error.localizedDescription }
            case .failure(let e): error = e.localizedDescription
            }
        }
    }
}

extension ContentView {
    /// Debug builds only: HAWKEYE_OPEN="review:<session id>" (or calibrate / analyze /
    /// landing:<id>:<frame>) opens that screen at launch, for testing without tapping.
    func openFromEnvironment() {
        #if DEBUG
        guard path.isEmpty, let v = ProcessInfo.processInfo.environment["HAWKEYE_OPEN"] else { return }
        if v == "match" { showMatch = true; return }
        let p = v.split(separator: ":").map(String.init)
        guard p.count >= 2 else { return }
        switch p[0] {
        case "calibrate": path = [.session(p[1]), .calibrate(p[1])]
        case "analyze": path = [.session(p[1]), .analyze(p[1])]
        case "review": path = [.session(p[1]), .review(p[1])]
        case "landing" where p.count >= 3: path = [.session(p[1]), .review(p[1]), .landing(p[1], Int(p[2]) ?? 0, 1)]
        default: path = [.session(p[1])]
        }
        #endif
    }
}

struct SessionRow: View {
    let s: SessionRecord
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(s.created.formatted(date: .abbreviated, time: .shortened))
            if let a = s.analysis {
                let c = a.landings.map(\.result.call)
                Text("\(a.landings.count) 个落点 · 界内 \(c.filter { $0 == .IN }.count) · 出界 \(c.filter { $0 == .OUT }.count) · 判不了 \(c.filter { $0 == .ABSTAIN }.count)")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                Text(s.calibrationPoints.count >= 4 ? "已标定，未分析" : "未标定").font(.caption).foregroundStyle(.orange)
            }
        }
    }
}

/// One session: calibrate -> analyse -> review.
struct SessionView: View {
    @EnvironmentObject var store: Store
    let id: String
    var body: some View {
        if let s = store.sessions.first(where: { $0.id == id }) {
            List {
                Section("1 标定场地") {
                    NavigationLink(value: Route.calibrate(id)) {
                        Label(s.calibrationPoints.count >= 4 ? "已标 \(s.calibrationPoints.count) 个点（可修改）" : "在画面上点场地线交点",
                              systemImage: s.calibrationPoints.count >= 4 ? "checkmark.circle.fill" : "scope")
                    }
                    Picker("比赛", selection: Binding(get: { s.mode }, set: { var t = s; t.mode = $0; t.analysis = nil; store.save(t) })) {
                        ForEach(Mode.allCases, id: \.self) { Text($0.cn).tag($0) }
                    }
                }
                Section("2 找落点并判界") {
                    NavigationLink(value: Route.analyze(id)) {
                        Label(s.analysis == nil ? "开始分析" : "重新分析", systemImage: "wand.and.stars")
                    }.disabled(s.calibrationPoints.count < 4)
                }
                if let a = s.analysis {
                    Section("3 回看") {
                        NavigationLink(value: Route.review(id)) {
                            Label("\(a.landings.count) 个落点（另有 \(a.nonRally.count) 个不像回合落点）", systemImage: "play.rectangle")
                        }
                    }
                }
                Section {
                    Text("文件位置：文件 App › 我的 iPhone › 手机鹰眼 › Sessions › \(s.id)").font(.caption).foregroundStyle(.secondary)
                }
            }
            .navigationTitle(s.created.formatted(date: .abbreviated, time: .shortened))
        } else {
            Text("记录不存在")
        }
    }
}
