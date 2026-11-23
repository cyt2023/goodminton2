import AVFoundation
import Foundation
import SwiftUI

/// Sessions on disk: Documents/Sessions/<id>/{video.mov, session.json}.
/// Documents is visible in the Files app (iPhone > 手机鹰眼), so a session folder
/// can be AirDropped to the Mac.
@MainActor
final class Store: ObservableObject {
    @Published var sessions: [SessionRecord] = []
    let root: URL

    init() {
        root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("Sessions", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        load()
    }

    static let encoder: JSONEncoder = {
        let e = JSONEncoder(); e.outputFormatting = [.prettyPrinted, .sortedKeys]; e.dateEncodingStrategy = .iso8601; return e
    }()
    static let decoder: JSONDecoder = { let d = JSONDecoder(); d.dateDecodingStrategy = .iso8601; return d }()

    func folder(_ s: SessionRecord) -> URL { root.appendingPathComponent(s.id, isDirectory: true) }
    func videoURL(_ s: SessionRecord) -> URL { folder(s).appendingPathComponent(s.videoFile) }

    func load() {
        let dirs = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        sessions = dirs.compactMap { d in
            guard let data = try? Data(contentsOf: d.appendingPathComponent("session.json")) else { return nil }
            return try? Store.decoder.decode(SessionRecord.self, from: data)
        }.sorted { $0.created > $1.created }
    }

    func save(_ s: SessionRecord) {
        do {
            try FileManager.default.createDirectory(at: folder(s), withIntermediateDirectories: true)
            try Store.encoder.encode(s).write(to: folder(s).appendingPathComponent("session.json"), options: .atomic)
            if let i = sessions.firstIndex(where: { $0.id == s.id }) { sessions[i] = s } else { sessions.insert(s, at: 0) }
        } catch {
            print("save failed: \(error)")
        }
    }

    /// New session from a video file (moved if it is a temporary recording, copied otherwise).
    func create(fromVideo url: URL, move: Bool) throws -> SessionRecord {
        let fmt = DateFormatter(); fmt.dateFormat = "yyyyMMdd-HHmmss"
        let id = fmt.string(from: Date())
        let ext = url.pathExtension.isEmpty ? "mov" : url.pathExtension.lowercased()
        var s = SessionRecord(id: id, created: Date(), videoFile: "video.\(ext)")
        try FileManager.default.createDirectory(at: folder(s), withIntermediateDirectories: true)
        if move { try FileManager.default.moveItem(at: url, to: videoURL(s)) } else { try FileManager.default.copyItem(at: url, to: videoURL(s)) }
        s.calibrationFrameTime = 1.0
        save(s)
        return s
    }

    func delete(_ s: SessionRecord) {
        try? FileManager.default.removeItem(at: folder(s))
        sessions.removeAll { $0.id == s.id }
    }
}
