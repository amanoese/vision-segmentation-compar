# 固定Queryの由来

元の目的は「肌露出」です。2026-10-16の同系列の装甲破裂キャラ画像を使った公式SAM Agent実験では、
LLMが `figurine` を試した後、`exposed torso` をSAMへのQueryに選びました。
今回の動画では、その文字列を全145フレームで固定して使用しています。

| 項目 | 値 |
| --- | --- |
| 過去のAgentへの元指示 | 肌の露出が多い部分 |
| 今回の元指示 | 肌露出 |
| 再利用する検出Query | `exposed torso` |
| 過去の判断役LLM | `meta-llama/llama-4-maverick` |
| 検出モデル | `fal-ai/sam-3-1/image` |
| 過去の公式SAM Agentコミット | `7f473cfd983d5eb43040ff608f66bff0d7bdcb53` |
| 参照画像のSHA-256 | `ff325efa936d189d5d16f151470ceb7119f44569c806a0ff94f3ee467babee44` |
| 今回の追加LLM呼出し | 0 |

過去の指示と今回の指示は同一文言ではありません。
この列は「毎フレームでAgentが確認・選択した結果」ではなく、「Agent由来Queryを固定したSAM 3.1」の結果です。
Queryが露出した胴体を指すため、腕や手は対象から外れる場合があります。

過去の実験元は旧PoCの `fal-sam31-agent-skin-image-2026-10-16` です。
生のLLM履歴や公式ソース一式はGitには複製せず、必要な条件だけをここに記録しています。
