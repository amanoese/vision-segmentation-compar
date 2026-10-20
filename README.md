# vision-segmentation-compare

生成したフィギュア動画で、SAM 3.1・SAM Agent・Jevの検出とモザイク処理を比較する、ブログ用の実装リポジトリです。
肌が見える動画と内部がメカの動画を使い、`figure`、SNS向けの不適切表現、露出した肌の部位を検出します。

[結果・完成動画・GIFを見る](results/README.md)

## 公開するファイル

```text
src/                 API連携、SAM Agent、フレーム処理、描画、CLI
configs/             ブログの比較に使用した5つの設定
  skin-figure-openrouter-agent.json
  skin-sns-openrouter-agent.json
  skin-visible-parts.json
  mechanical-figure-openrouter-agent.json
  mechanical-sns-openrouter-agent.json
data/inputs/         入力動画2本（約9.3 MiB）
results/             完成動画・GIF・代表画像を3セット、集計とハッシュ
                       {skin,visible-skin,mechanical}.{mp4,gif,png}
tests/               APIをモックしたテスト
```

入力はどちらも1280×720、24fps、145フレーム、約6.04秒です。
入力動画2本と、完成版の比較MP4・GIF・代表画像を各3件Gitに含めます。生成履歴、API応答、判定キャッシュ、抽出フレーム、途中版や未使用の動画・GIF、過去の実験資料はローカルに残し、Gitから除外します。
`runs/`、`data/generated/`、`local-archive/` はGit対象外です。公開する動画・GIFはファイル単位で許可しています。

## 実装の流れ

```text
入力MP4 → FFmpegでフレーム抽出 → 各方式で検出 → bboxを保存
        → Sharpで検出枠・モザイクを描画 → FFmpegでMP4に結合
```

- **SAM**：元のQueryをfal.aiのSAM 3.1へ送り、候補からbboxを取得。
- **SAM Agent**：OpenRouterのLLMがフレームを見てQueryを具体化し、SAMの候補マスクを確認。1フレームでLLM最大3回、SAM最大2回。
- **Jev**：存在確率とbboxを1リクエストで取得。存在確率0.8以上、bbox信頼度0.6以上を採用。

SAM Agentは[SAM 3のAgent設計](https://arxiv.org/html/2511.16719v1#A7.SS1)を参考にした独自実装です。公式実装の完全な再現ではなく、呼び出し回数の制限もこの実験の設定です。

主な処理は [src/video.ts](src/video.ts)、Agentは [src/agent/samAgent.ts](src/agent/samAgent.ts)、モザイクは [src/render.ts](src/render.ts)、ブログの2条件重ね描画は [src/renderDualQueryBbox.ts](src/renderDualQueryBbox.ts) にあります。
モザイクはbboxの矩形内に適用します。ライブ配信やフレーム間の追跡は実装していません。

## セットアップ

Node.js 24、pnpm 11.5.2、FFmpeg/ffprobeを使用します。

```bash
pnpm install --frozen-lockfile
cp .env.example .env
```

新たに検出する場合は `.env` に `JEV_API_KEY`、`FALAI_API_KEY`、`OPENROUTER_API_KEY` を設定します。
デフォルトのAgentモデルは `qwen/qwen3.7-flash` です。

## 検出

次の例は肌動画のSNS条件です。APIを呼び出し、利用料金が発生します。
設定ファイルの `input` と `runDir` は設定ファイルからの相対パスです。

```bash
pnpm prepare:video --config configs/skin-sns-openrouter-agent.json
pnpm detect --config configs/skin-sns-openrouter-agent.json --method sam
pnpm detect --config configs/skin-sns-openrouter-agent.json --method sam-agent
pnpm detect --config configs/skin-sns-openrouter-agent.json --method jev
```

残りの4設定も同じ手順で処理します。成功済みの判定はキャッシュから再利用します。
入力・Query・モデルを変更するときは別の `runDir` を指定してください。

## ブログの比較動画を生成

各ペアの全方式・全フレームの判定が揃ってから実行します。
描画はAPIを呼びません。Gitには判定キャッシュを含めないため、新規cloneでは先に検出が必要です。

```bash
# 肌：figure（青）＋SNS条件（赤）
pnpm render:blog configs/skin-figure-openrouter-agent.json configs/skin-sns-openrouter-agent.json runs/blog-skin-comparison/outlined-mosaic --mosaic

# 肌：figure（青）＋露出部位（赤）
pnpm render:blog configs/skin-figure-openrouter-agent.json configs/skin-visible-parts.json runs/blog-visible-skin-comparison/outlined-mosaic --skin-target --mosaic

# メカ：figure（青）＋SNS条件（赤）
pnpm render:blog configs/mechanical-figure-openrouter-agent.json configs/mechanical-sns-openrouter-agent.json runs/blog-mechanical-comparison/outlined-mosaic --mosaic
```

出力先に比較MP4、代表PNG、集計JSONが生成されます。モザイクのセル幅は元映像で約48pxです。
`results/` には完成版の比較MP4と、そのGIF・代表PNGを選んで保存しています。GIFは高さ320px・20fpsの掲載用軽量版です。
単一条件の比較は `pnpm render:2x2 configs/skin-sns-openrouter-agent.json` で生成できます。

## 確認

```bash
pnpm typecheck
pnpm test
pnpm build
```

テストは実APIを呼びません。動画テストにはFFmpegが必要です。
検出フレーム数は正解率ではなく、SNS規約の適否や肌の全領域を隠せることを保証する評価ではありません。
