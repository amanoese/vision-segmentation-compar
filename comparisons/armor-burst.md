# 装甲破裂動画の比較

2026-10-19検証。1280×720、24fps、145フレーム、6.041667秒。

[6パネル動画](../runs/armor-burst/comparison/comparison.mp4) ·
[3秒の比較画像](../runs/armor-burst/comparison/representatives/000073.png) ·
[5秒の比較画像](../runs/armor-burst/comparison/representatives/000121.png)

左からSAM 3.1、SAM Agent（SAM 3.1・調整Query固定）、JEV。
上段は青枠＋内側25%の半透明青塗り、下段は同じbbox領域のモザイクです。

| 方式 | 検出Query | bbox採用フレーム |
| --- | --- | ---: |
| SAM 3.1単体 | 肌露出 | 0 / 145 |
| SAM Agent調整Query固定 | `exposed torso` | 120 / 145 |
| JEV | 肌露出 | 78 / 145 |

SAM 3.1単体の日本語Queryはこの素材では検出なしでした。英語Queryへ調整した列では胸・腹を中心に検出しています。
JEVでは背景やヘルメットの混入、信頼度不足による枠の消失が見られます。
検出件数は精度ではなく、正解annotationを使った定量評価はしていません。

移行ではモデルを再実行せず、保存済みの3方式の判定から比較動画を再生成しています。
Queryの由来と方式の違いは [query-provenance.md](../docs/query-provenance.md) を参照してください。
