# vision-segmentation-compare

同じ動画の全フレームで、**SAM 3.1 → SAM Agent（SAM 3.1・調整Query固定）→ JEV**を比較するPoCです。
上段はbboxの青枠と内側25%の半透明塗り、下段は同じbbox内のモザイクです。

SAM Agent欄は、以前のAgent実験で選んだ `exposed torso` を全フレームで固定してSAM 3.1に渡します。
毎フレームのLLM判断は行いません。この比較にはOpenRouterのAPIキーは不要です。
JEVとSAM 3.1単体には日本語の「肌露出」をそのまま渡します。共通の目的に対する、直接入力とAgent調整Queryの比較です。

## セットアップ

Node.js 24、pnpm 11.5.2、FFmpeg/ffprobeが必要です。

```bash
pnpm install --frozen-lockfile
cp .env.example .env
# 新たな推論を行う場合だけ .env に JEV_API_KEY と FALAI_API_KEY を設定
```

設定は [configs/armor-burst.json](configs/armor-burst.json) です。
`input` と `runDir` は設定ファイルからの相対パスで、入力動画は `data/inputs/armor-burst.mp4` に配置します。
別の素材・Queryでは設定と `runDir` を分けてください。このPoCは固定fpsでフレーム数を取得できる動画を対象にします。

## 実行

今回の動画・判定データはGit対象外のローカルデータとして移行済みです。
**保存済みの判定からの再描画はAPIキー不要・API呼出し0回**です。

```bash
pnpm render
```

新しい推論は次のコマンドで実行します。各サービスへフレーム画像が送信され、API料金が発生します。
`detect` は保存済みのフレーム判定を再利用します。

```bash
pnpm prepare:video
pnpm detect --method sam
pnpm detect --method sam-agent-fixed
pnpm detect --method jev
pnpm render

# 3方式の検出と描画をまとめて実行
pnpm compare

# 別の設定を利用
pnpm compare --config configs/another-case.json
```

出力は `runs/armor-burst/comparison/` にあります。

- `comparison.mp4`: 全フレームの6パネル動画
- `representatives/`: 0〜6秒の代表画像。各時点のフォルダに方式別のbbox・モザイク画像も保存
- `frames/`: 比較動画の各フレーム。再生成可能なため今回の移行では省略
- `summary.json` / `detections.jsonl`: 件数、モデル条件、各フレームの判定

検証結果と出力へのリンクは [comparisons/armor-burst.md](comparisons/armor-burst.md) を参照してください。

## Gitに保存するもの

```text
src/            検出、画像処理、動画生成、CLI
tests/          APIを使わないテスト
configs/        入力・出力先とQuery
docs/           Queryの由来、データの配置方針
comparisons/    小さな比較記録と既存のresults.csv
package.json / pnpm-lock.yaml / tsconfig.json / .env.example
```

`data/`、`runs/`、`images/`の実画像、`node_modules/`、`.env`、ビルド成果物はGit対象外です。
元のClef実験、全API応答、古い動画、Python環境、毎フレームLLM版のコードはこのプロジェクトに含めていません。
既存の `images/` フォルダと `comparisons/results.csv` は残しています。
動画や入力データを他の環境へ持ち出す際は、Gitとは別に転送してください。

## 確認

```bash
pnpm typecheck
pnpm test
pnpm build
git status --short
```

テストのAPIはモックされており、実際のモデルには送信しません。動画テストではFFmpegを使用します。

## 比較上の制約

JEVは存在確率0.8以上・bbox信頼度0.6以上だけ採用します。SAM側は返された領域を使用します。
スコアは方式間で同じ意味ではありません。モザイクはマスク形状ではなく矩形bboxの和集合に適用し、時間方向の追跡や平滑化は行いません。
検出フレーム数は正解率ではなく、肌の全領域を網羅する保証もありません。
