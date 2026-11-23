# goodminton2 — 手机鹰眼（羽毛球落点判界）

用 iPhone 拍羽毛球场地，实时识别每一次落地，判断界内 / 出界 / 判不了（带误差范围），并可以回放上一个落点的截图和 3D 场地。支持单机，也支持两部手机各放在一端、合并两个视角来判。

| 目录 | 内容 |
|---|---|
| `ios/` | iPhone App（SwiftUI + Core ML）。安装和用法见 [ios/README.md](ios/README.md) |
| `prototype/` | Mac 上的 Python 原型：标定、落点检测、不确定性、场测标注与报告。见 [prototype/README.md](prototype/README.md) |
| `场测指南_v1.md` | 第一次场地测试的流程和要回答的问题 |

## 依赖

- **iPhone App**：需要 Xcode 16 或更新版本，打开 `ios/HawkEye.xcodeproj` 后，在 Signing 里选自己的 Apple ID 即可编译。TrackNetV3 的 Core ML 模型已经包含在 `ios/HawkEye/Resources/` 里。
- **`prototype/` 和模型转换脚本 `ios/tools/convert_tracknet.py`**：需要 TrackNetV3 的源码、权重和 Python 环境。它们在原作者的仓库里（[qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3)），不在本仓库中；脚本默认它们放在仓库旁边的 `04_原型/` 目录里。

## 第三方

TrackNetV3（MIT License），见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
