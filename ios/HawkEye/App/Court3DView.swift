import SceneKit
import SwiftUI
import UIKit

/// Hawk-Eye style 3D court: lines, net, the judged boundary, the measured landing
/// point with its 95% ellipse, and a shuttle flying in along an illustrative arc.
/// One camera cannot measure height, so only the landing point is measured.
struct Court3DView: UIViewRepresentable {
    let landing: Landing
    let mode: Mode
    var overview: Bool
    var replayToken: Int

    final class Coordinator { var key = "" }
    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> SCNView {
        let v = SCNView()
        v.backgroundColor = UIColor(red: 0.08, green: 0.1, blue: 0.12, alpha: 1)
        v.antialiasingMode = .multisampling4X
        v.allowsCameraControl = true
        v.autoenablesDefaultLighting = false
        return v
    }

    func updateUIView(_ v: SCNView, context: Context) {
        let key = "\(landing.id)-\(mode.rawValue)-\(overview)-\(replayToken)"
        guard key != context.coordinator.key else { return }
        context.coordinator.key = key
        v.scene = Self.scene(landing: landing, mode: mode, overview: overview)
        v.pointOfView = v.scene?.rootNode.childNode(withName: "camera", recursively: false)
    }

    /// Court metres -> scene: x across, y up, near end (court y = 0) towards +z.
    static func pos(_ p: Pt, _ h: Double = 0) -> SCNVector3 {
        SCNVector3(Float(p.x - Court.width / 2), Float(h), Float(Court.length / 2 - p.y))
    }

    static func material(_ c: UIColor, emission: Bool = false, alpha: CGFloat = 1) -> SCNMaterial {
        let m = SCNMaterial()
        m.diffuse.contents = c
        if emission { m.emission.contents = c }
        m.transparency = alpha
        m.isDoubleSided = true
        m.lightingModel = .physicallyBased
        return m
    }

    static func box(from a: Pt, to b: Pt, width: Double, height: Double, y: Double, color: UIColor, emission: Bool = false) -> SCNNode {
        let dx = b.x - a.x, dy = b.y - a.y
        let len = (dx * dx + dy * dy).squareRoot()
        let g = SCNBox(width: CGFloat(abs(dx) > abs(dy) ? len : width), height: CGFloat(height),
                       length: CGFloat(abs(dx) > abs(dy) ? width : len), chamferRadius: 0)
        g.materials = [material(color, emission: emission)]
        let n = SCNNode(geometry: g)
        n.position = pos(Pt((a.x + b.x) / 2, (a.y + b.y) / 2), y)
        return n
    }

    /// Flat strip between two court points at any angle.
    static func segment(from a: Pt, to b: Pt, width: Double, y: Double, color: UIColor) -> SCNNode {
        let d = b - a
        let g = SCNBox(width: CGFloat(d.norm + width * 0.5), height: 0.002, length: CGFloat(width), chamferRadius: 0)
        g.materials = [material(color, emission: true)]
        let n = SCNNode(geometry: g)
        n.position = pos(Pt((a.x + b.x) / 2, (a.y + b.y) / 2), y)
        n.eulerAngles.y = Float(atan2(d.y, d.x))   // court angle -> rotation about +y (z = -court y)
        return n
    }

    static func scene(landing l: Landing, mode: Mode, overview: Bool) -> SCNScene {
        let scene = SCNScene()
        let root = scene.rootNode
        let callColor: UIColor = l.result.call == .IN ? .systemGreen : (l.result.call == .OUT ? .systemRed : .systemOrange)

        // Floor and court surface.
        let floor = SCNPlane(width: 24, height: 32)
        floor.materials = [material(UIColor(red: 0.16, green: 0.2, blue: 0.22, alpha: 1))]
        let fn = SCNNode(geometry: floor); fn.eulerAngles.x = -.pi / 2; fn.position.y = -0.002
        root.addChildNode(fn)
        let surface = SCNPlane(width: Court.width + 2, height: Court.length + 3)
        surface.materials = [material(UIColor(red: 0.13, green: 0.45, blue: 0.33, alpha: 1))]
        let sn = SCNNode(geometry: surface); sn.eulerAngles.x = -.pi / 2
        root.addChildNode(sn)

        // Painted lines (centre lines, 40 mm wide) and the boundary used for the call.
        for (a, b) in Court.lineSegments() { root.addChildNode(box(from: a, to: b, width: Court.line, height: 0.002, y: 0.001, color: .white)) }
        let bd = Court.boundary(mode)
        let c = [Pt(bd.x0, bd.y0), Pt(bd.x1, bd.y0), Pt(bd.x1, bd.y1), Pt(bd.x0, bd.y1), Pt(bd.x0, bd.y0)]
        for i in 0..<4 { root.addChildNode(box(from: c[i], to: c[i + 1], width: 0.012, height: 0.003, y: 0.003, color: .systemYellow, emission: true)) }

        // Net: posts on the doubles sidelines, 1.55 m; net 0.76 m deep; white tape on top.
        for x in [-0.02, Court.width + 0.02] {
            let post = SCNCylinder(radius: 0.02, height: 1.55); post.materials = [material(.darkGray)]
            let pn = SCNNode(geometry: post); pn.position = pos(Pt(x, Court.netY), 0.775); root.addChildNode(pn)
        }
        let net = SCNPlane(width: Court.width, height: 0.76); net.materials = [material(.white, alpha: 0.25)]
        let nn = SCNNode(geometry: net); nn.position = pos(Pt(Court.width / 2, Court.netY), 1.55 - 0.38); root.addChildNode(nn)
        root.addChildNode(box(from: Pt(0, Court.netY), to: Pt(Court.width, Court.netY), width: 0.02, height: 0.04, y: 1.53, color: .white))

        // Landing: 95% ellipse (measured), cork footprint and a ring to find it.
        // Ellipse as a 2 cm outline plus a faint fill, so the line and the shuttle stay visible
        // even when the ellipse is long (large uncertainty along the camera's line of sight).
        let centre = l.result.courtXY
        let outer = Uncertainty.ellipse(centre: centre, cov: l.result.cov, steps: 96).map { $0 - centre }
        let a = outer.map(\.norm).max() ?? 0.1
        let inner = outer.map { $0 * max(0, 1 - 0.02 / max(a, 0.02)) }
        func shapePath(_ pts: [Pt]) -> UIBezierPath {
            let p = UIBezierPath(); p.move(to: CGPoint(x: pts[0].x, y: pts[0].y))
            pts.dropFirst().forEach { p.addLine(to: CGPoint(x: $0.x, y: $0.y)) }; p.close(); return p
        }
        _ = inner
        let fillPath = shapePath(outer); fillPath.flatness = 0.001
        let fill = SCNShape(path: fillPath, extrusionDepth: 0.001)
        let fm = SCNMaterial()
        fm.diffuse.contents = callColor.withAlphaComponent(0.18)
        fm.lightingModel = .constant
        fm.blendMode = .alpha
        fm.transparencyMode = .aOne
        fm.writesToDepthBuffer = false
        fm.isDoubleSided = true
        fill.materials = [fm]
        let fn2 = SCNNode(geometry: fill)
        fn2.eulerAngles.x = -.pi / 2            // shape (dx, dy) -> scene (dx, 0, -dy), matching pos()
        fn2.position = pos(centre, 0.004)
        root.addChildNode(fn2)
        for i in 0..<(outer.count - 1) {          // outline from short segments
            root.addChildNode(segment(from: centre + outer[i], to: centre + outer[i + 1], width: 0.02, y: 0.005, color: callColor))
        }
        let cork = SCNCylinder(radius: 0.0135, height: 0.004); cork.materials = [material(callColor, emission: true)]
        let cn = SCNNode(geometry: cork); cn.position = pos(l.result.courtXY, 0.007); root.addChildNode(cn)
        let ring = SCNTorus(ringRadius: 0.12, pipeRadius: 0.006); ring.materials = [material(callColor, emission: true)]
        let rn = SCNNode(geometry: ring); rn.position = pos(l.result.courtXY, 0.01); root.addChildNode(rn)
        rn.runAction(.repeatForever(.sequence([.scale(to: 1.6, duration: 0.6), .scale(to: 1.0, duration: 0.6)])))

        // Shuttle: cork + skirt, flying in from the other side along an illustrative arc.
        let shuttle = SCNNode()
        let corkBall = SCNSphere(radius: 0.017); corkBall.materials = [material(.white)]
        shuttle.addChildNode(SCNNode(geometry: corkBall))
        let skirt = SCNCone(topRadius: 0.014, bottomRadius: 0.034, height: 0.065); skirt.materials = [material(.white, alpha: 0.85)]
        let sk = SCNNode(geometry: skirt); sk.position.y = 0.045; shuttle.addChildNode(sk)
        root.addChildNode(shuttle)
        let land = l.result.courtXY
        let startY = land.y < Court.netY ? min(Court.length - 1, Court.netY + 4) : max(1, Court.netY - 4)
        let start = Pt(land.x * 0.6 + Court.width / 2 * 0.4, startY)
        let path: [SCNVector3] = (0...40).map { i in
            let t = Double(i) / 40
            let p = start + (land - start) * (1 - pow(1 - t, 1.6))          // decelerating, as with drag
            let h = 2.2 * (1 - t) + 1.4 * sin(.pi * t) * (1 - t)             // high, then a steep drop
            return pos(p, max(h, 0) + 0.017)
        }
        for p in path.enumerated() where p.offset % 2 == 0 {
            let dot = SCNSphere(radius: 0.008); dot.materials = [material(.systemOrange, emission: true, alpha: 0.6)]
            let dn = SCNNode(geometry: dot); dn.position = p.element; root.addChildNode(dn)
        }
        shuttle.position = path[0]
        var steps: [SCNAction] = [.wait(duration: 0.4)]
        for p in path.dropFirst() { steps.append(.move(to: p, duration: 1.4 / 40)) }
        steps.append(.wait(duration: 1.8))
        steps.append(.move(to: path[0], duration: 0))
        shuttle.runAction(.repeatForever(.sequence(steps)))
        shuttle.eulerAngles.x = 0.5

        // Lights.
        let amb = SCNLight(); amb.type = .ambient; amb.intensity = 600
        let an = SCNNode(); an.light = amb; root.addChildNode(an)
        let sun = SCNLight(); sun.type = .directional; sun.intensity = 900; sun.castsShadow = true
        let sunN = SCNNode(); sunN.light = sun; sunN.eulerAngles = SCNVector3(-1.0, 0.4, 0); root.addChildNode(sunN)

        // Camera: low and close to the landing, looking along the nearest line, or an overview.
        let cam = SCNCamera(); cam.fieldOfView = overview ? 50 : 38; cam.zNear = 0.01
        let camN = SCNNode(); camN.camera = cam; camN.name = "camera"
        let target = pos(land, 0)
        if overview {
            camN.position = SCNVector3(0, 13, Float(Court.length / 2) + 5)
            camN.look(at: SCNVector3(0, 0, 0))
        } else {
            // Hawk-Eye style: low, outside the nearest boundary line, looking back across the
            // line at the landing, slightly from the side so the line runs across the picture.
            let outward: Pt = ["left": Pt(-1, 0), "right": Pt(1, 0), "near": Pt(0, -1), "far": Pt(0, 1)][l.result.nearestEdge.rawValue]!
            let along = Pt(-outward.y, outward.x)
            let eye = land + outward * 1.3 + along * 0.6
            camN.position = pos(eye, 0.45)
            camN.look(at: pos(land, 0.02))
        }
        root.addChildNode(camN)
        return scene
    }
}

/// Replay of the landings of a match: snapshot of the contact frame + 3D view.
struct ReplayView: View {
    let folder: URL?
    let landings: [Landing]
    let mode: Mode
    @State var index: Int
    @State private var overview = false
    @State private var token = 0
    @Environment(\.dismiss) private var dismiss

    @ViewBuilder private func snapshot(_ f: String?) -> some View {
        if let f, let folder, let img = UIImage(contentsOfFile: folder.appendingPathComponent(f).path) {
            Image(uiImage: img).resizable().scaledToFit().clipShape(RoundedRectangle(cornerRadius: 8))
        } else {
            RoundedRectangle(cornerRadius: 8).fill(.gray.opacity(0.2)).overlay(Text("没有截图"))
        }
    }

    var body: some View {
        let l = landings[min(max(index, 0), landings.count - 1)]
        VStack(spacing: 0) {
            HStack {
                Button { dismiss() } label: { Label("返回比赛", systemImage: "chevron.left") }
                Spacer()
                Text("落点 \(index + 1) / \(landings.count)" + ((l.views?.count ?? 0) == 2 ? " · 两机合并" : "")).font(.headline)
                Spacer()
                Button { index -= 1 } label: { Image(systemName: "chevron.left.circle.fill").font(.title2) }.disabled(index == 0)
                Button { index += 1 } label: { Image(systemName: "chevron.right.circle.fill").font(.title2) }.disabled(index >= landings.count - 1)
            }
            .padding(.horizontal).padding(.vertical, 8)
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 6) {
                    if let views = l.views, views.count == 2 {
                        // Two phones: each one's snapshot and its own call; the big call is the combination.
                        ForEach(views, id: \.side) { v in
                            ZStack(alignment: .topLeading) {
                                snapshot(v.snapshotFile)
                                Text("\(v.side) 端：\(v.result.call.cn) · ±\(Int(v.result.distanceSigma * 1000)) mm").font(.caption2.bold())
                                    .padding(4).background(.black.opacity(0.6)).foregroundStyle(.white).padding(4)
                            }
                        }
                    } else {
                        snapshot(l.snapshotFile)
                    }
                    Text(String(format: "触地 %.1f s · 离%@ %+.0f ± %.0f mm（正 = 界内）", l.candidate.contactTime, l.result.nearestEdge.cn,
                                l.result.signedDistance * 1000, l.result.distanceSigma * 1000)).font(.caption)
                    if let r = l.result.abstainReason { Text("判不了：\(r)").font(.caption).foregroundStyle(.orange) }
                }
                .frame(maxWidth: .infinity)
                ZStack(alignment: .top) {
                    Court3DView(landing: l, mode: mode, overview: overview, replayToken: token)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    VStack {
                        Text(l.result.call.cn).font(.system(size: 34, weight: .heavy)).foregroundStyle(.white)
                            .padding(.horizontal, 18).padding(.vertical, 4).background(l.result.call.color, in: Capsule())
                        Text(String(format: "P(界内) = %.2f", l.result.pIn)).font(.caption.bold()).foregroundStyle(.white)
                        Spacer()
                        HStack {
                            Text("轨迹为示意，只有落点和误差椭圆是测量值").font(.caption2).foregroundStyle(.white.opacity(0.8))
                            Spacer()
                            Picker("", selection: $overview) { Text("近看").tag(false); Text("全场").tag(true) }.pickerStyle(.segmented).frame(width: 120)
                            Button { token += 1 } label: { Image(systemName: "arrow.counterclockwise") }.buttonStyle(.bordered)
                        }
                        .padding(8)
                    }
                    .padding(.top, 8)
                }
                .frame(maxWidth: .infinity)
            }
            .padding(.horizontal, 8).padding(.bottom, 8)
        }
        .background(Color(.systemBackground))
    }
}
