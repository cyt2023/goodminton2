import Foundation
import MultipeerConnectivity
import QuartzCore
import UIKit

enum DuoRole: String, CaseIterable {
    case solo, host, guest
    var cn: String { ["solo": "单机", "host": "双机 A 端（主机）", "guest": "双机 B 端"][rawValue]! }
}

/// Messages between the two phones (JSON over Multipeer Connectivity).
struct DuoMessage: Codable {
    enum Kind: String, Codable { case ping, pong, status, start, stop, landing }
    var kind: Kind
    var t0: Double? = nil, t1: Double? = nil          // clock sync
    var calibrated: Bool? = nil, rms: Double? = nil   // status
    var mode: Mode? = nil                              // start
    var landing: Landing? = nil                        // in the HOST's court coordinates
    var time: Double? = nil                            // contact time, HOST clock
    var snapshot: Data? = nil                          // JPEG of the guest's contact frame
    var zoom: Data? = nil                              // JPEG full-resolution crop around the shuttle
}

/// Pairs two phones directly (Wi-Fi / Bluetooth, no router or internet needed).
/// A (host) advertises; B (guest) finds it and joins. After connecting, B measures the
/// offset between the two clocks with ping/pong messages (minimum round trip of 15).
/// Camera timestamps use the same clock as CACurrentMediaTime(), so B can convert its
/// landing times to A's clock.
final class DualLink: NSObject, ObservableObject, MCSessionDelegate, MCNearbyServiceAdvertiserDelegate, MCNearbyServiceBrowserDelegate {
    static let service = "hawkeye-duo"
    @Published private(set) var role: DuoRole = .solo
    @Published private(set) var connected = false
    @Published private(set) var peerName: String?
    @Published var peerCalibrated = false
    @Published var peerRMS: Double?
    @Published private(set) var clockOffset: Double?   // host clock minus this phone's clock (guest)
    @Published private(set) var roundTrip: Double?
    var onMessage: ((DuoMessage) -> Void)?

    private let me = MCPeerID(displayName: UIDevice.current.name)
    private var session: MCSession?
    private var advertiser: MCNearbyServiceAdvertiser?
    private var browser: MCNearbyServiceBrowser?
    private var pings = 0

    static func now() -> Double { CACurrentMediaTime() }

    func start(_ r: DuoRole) {
        stop()
        role = r
        guard r != .solo else { return }
        let s = MCSession(peer: me, securityIdentity: nil, encryptionPreference: .required)
        s.delegate = self
        session = s
        if r == .host {
            let a = MCNearbyServiceAdvertiser(peer: me, discoveryInfo: ["role": "A"], serviceType: Self.service)
            a.delegate = self; a.startAdvertisingPeer(); advertiser = a
        } else {
            let b = MCNearbyServiceBrowser(peer: me, serviceType: Self.service)
            b.delegate = self; b.startBrowsingForPeers(); browser = b
        }
    }

    func stop() {
        advertiser?.stopAdvertisingPeer(); advertiser = nil
        browser?.stopBrowsingForPeers(); browser = nil
        session?.disconnect(); session = nil
        connected = false; peerName = nil; peerCalibrated = false; peerRMS = nil; clockOffset = nil; roundTrip = nil
        role = .solo
    }

    func send(_ m: DuoMessage) {
        guard let s = session, !s.connectedPeers.isEmpty, let data = try? JSONEncoder().encode(m) else { return }
        try? s.send(data, toPeers: s.connectedPeers, with: .reliable)
    }

    /// Guest -> host clock.
    func toHostClock(_ t: Double) -> Double { role == .guest ? t + (clockOffset ?? 0) : t }

    private func syncClock() {
        pings = 0
        for i in 0..<15 {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1 * Double(i)) { [weak self] in
                self?.send(DuoMessage(kind: .ping, t0: Self.now()))
            }
        }
    }

    // MARK: delegates (called on Multipeer's queues; state is published on main)

    func advertiser(_ advertiser: MCNearbyServiceAdvertiser, didReceiveInvitationFromPeer peerID: MCPeerID, withContext context: Data?,
                    invitationHandler: @escaping (Bool, MCSession?) -> Void) {
        let s = session
        DispatchQueue.main.async { invitationHandler(s != nil && !(s!.connectedPeers.count > 0), s) }
    }

    func browser(_ browser: MCNearbyServiceBrowser, foundPeer peerID: MCPeerID, withDiscoveryInfo info: [String: String]?) {
        guard info?["role"] == "A", let s = session else { return }
        browser.invitePeer(peerID, to: s, withContext: nil, timeout: 15)
    }
    func browser(_ browser: MCNearbyServiceBrowser, lostPeer peerID: MCPeerID) {}

    func session(_ session: MCSession, peer peerID: MCPeerID, didChange state: MCSessionState) {
        DispatchQueue.main.async { [self] in
            connected = state == .connected
            peerName = state == .connected ? peerID.displayName : nil
            if state == .connected && role == .guest { syncClock() }
            if state == .notConnected { peerCalibrated = false }
        }
    }

    func session(_ session: MCSession, didReceive data: Data, fromPeer peerID: MCPeerID) {
        let t = Self.now()
        guard let m = try? JSONDecoder().decode(DuoMessage.self, from: data) else { return }
        switch m.kind {
        case .ping:
            send(DuoMessage(kind: .pong, t0: m.t0, t1: t))
        case .pong:
            guard let t0 = m.t0, let t1 = m.t1 else { return }
            let rtt = t - t0, off = t1 - (t0 + t) / 2
            DispatchQueue.main.async { [self] in
                if roundTrip == nil || rtt < roundTrip! { roundTrip = rtt; clockOffset = off }
            }
        case .status:
            DispatchQueue.main.async { [self] in peerCalibrated = m.calibrated ?? false; peerRMS = m.rms }
        default:
            DispatchQueue.main.async { [self] in onMessage?(m) }
        }
    }

    func session(_ session: MCSession, didReceive stream: InputStream, withName streamName: String, fromPeer peerID: MCPeerID) {}
    func session(_ session: MCSession, didStartReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, with progress: Progress) {}
    func session(_ session: MCSession, didFinishReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, at localURL: URL?, withError error: Error?) {}
}
