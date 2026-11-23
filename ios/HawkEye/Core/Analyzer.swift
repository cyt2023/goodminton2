import Foundation

/// One analysed landing as stored in a session.
struct Landing: Codable, Identifiable {
    var id: Int { candidate.contactFrame }
    var candidate: LandingCandidate
    var result: LandingResult
    var pixelSigma: Double
    /// Research log: what the player called ("in" / "out" / "unsure"), and whether it was a real landing.
    var playerCall: String?
    var notALanding: Bool?
    /// Live matches: frame at contact with the overlay drawn, saved in the session folder.
    var snapshotFile: String? = nil
    /// Two-phone matches: each phone's own estimate (result above is the combination).
    var views: [LandingView]? = nil
}

struct Analysis: Codable {
    var landings: [Landing]            // rally landings on the floor
    var nonRally: [Landing]            // juggles, drops, pick-ups: shown separately
    var rejected: Int                  // resting points above the horizon or far outside the court
    var fps: Double
    var videoSize: [Double]
    var detectedFrames: Int, totalFrames: Int
    var calibrationRMS: Double
}

enum Analyzer {
    struct Settings {
        var mode: Mode = .singles
        var alpha = 0.05
        var detSigma = 6.0            // px: TrackNet gives the shuttle centre, not the cork contact (see README)
        var contactRadius = 0.013     // m: cork radius, so a touch of the line counts as in
        var maxOutside = 3.0
    }

    static func run(track: ShuttleTracker.Track, calibration cal: Calibration, settings s: Settings = Settings()) -> Analysis {
        let cands = LandingDetector.detect(track.xy, fps: track.fps, width: track.width)
        var landings: [Landing] = [], nonRally: [Landing] = [], rejected = 0
        for c in cands {
            guard cal.onGround(c.contactPx) else { rejected += 1; continue }
            let g = cal.toCourt(c.contactPx)
            if Court.signedDistance(g, .doubles).d < -s.maxOutside { rejected += 1; continue }
            let cov = Uncertainty.pixelCov(detSigma: s.detSigma, velocity: c.velocityPxPerFrame)
            var cand = c
            if c.contactFrame < track.times.count { cand.contactTime = track.times[c.contactFrame] }
            let r = Uncertainty.locate(cal, pixel: c.contactPx, cov: cov, mode: s.mode, contactRadius: s.contactRadius, alpha: s.alpha)
            let l = Landing(candidate: cand, result: r, pixelSigma: s.detSigma, playerCall: nil, notALanding: nil)
            if c.isRallyLanding { landings.append(l) } else { nonRally.append(l) }
        }
        return Analysis(landings: landings, nonRally: nonRally, rejected: rejected, fps: track.fps,
                        videoSize: [track.width, track.height], detectedFrames: track.xy.compactMap { $0 }.count,
                        totalFrames: track.xy.count, calibrationRMS: cal.residualRMS)
    }
}

/// A recording and everything derived from it, stored as session.json next to the video.
struct SessionRecord: Codable, Identifiable {
    var id: String
    var created: Date
    var videoFile: String               // file name inside the session folder
    var mode: Mode = .singles
    var calibrationPoints: [String: Pt] = [:]
    var calibrationFrameTime: Double = 1.0
    var analysis: Analysis?
    var note: String = ""
}
