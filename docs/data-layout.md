# データの配置と再現

Gitのソースと、ローカルの入力・成果物を分けます。入力・モデル条件が変わった場合は新しいrunDirを指定してください。
`protocol.json` が動画ハッシュとQueryを照合し、別条件での結果の混在を防ぎます。

```text
data/inputs/armor-burst.mp4          元の約6秒動画
runs/armor-burst/
  protocol.json                    動画ハッシュ、fps、モデルとQuery
  detections/{sam,sam-agent-fixed,jev}/000001.json
                                   最小限の判定キャッシュ
  source-frames/                   入力動画から自動抽出。移行時には省略
  raw/                             新たな推論時のAPI応答・マスク等
  comparison/
    comparison.mp4
    representatives/
    frames/                        再生成できるため移行時には省略
    summary.json
    detections.jsonl
```

今回の配置には元動画、全3方式の判定キャッシュ、完成動画、代表画像、実験条件を含めます。
何百枚もの再生成可能なフレーム、全マスク、失敗した旧Agent実験や他方式の過去結果は含めません。

別の環境へはGitのコードに加えて、`data/inputs/armor-burst.mp4` と
`runs/armor-burst/{protocol.json,detections/}` を転送すれば、APIを呼ばずに `pnpm render` で再現できます。
キャッシュがない素材で推論するとAPI料金が発生します。

途中のAPIエラーでは処理を停止し、成功済みの判定を残します。
`raw/` のみ存在して判定が未保存の場合、再送による重複課金を避けるため自動上書きしません。
保存済みjob/responseを確認して回復するか、別runDirで新たな試行を行ってください。
