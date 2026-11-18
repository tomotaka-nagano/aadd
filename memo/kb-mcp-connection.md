# kb-mcp 接続方式の修正

## 調査結果

- `../kb-mcp/src/protocol/server.py` は FastMCP の Streamable HTTP エンドポイント `/mcp` を提供する。
- `../kb-mcp/src/protocol/tools.py` は `search_documents` と `download_document` を MCP ツールとして登録する。
- aadd が仮定していた `/api/search_documents` と `/api/download_document` は存在しない。
- 検索結果のフィールドとダウンロードの text/base64 形式は既存処理と一致する。

## 方針

1. kb-mcp に REST API を追加せず、aadd を公開済みの MCP 契約へ合わせる。
2. 公式 MCP SDK で初期化・ツール呼出し・セッション終了を管理する。
3. 検索・原文取得の既存動作と外部呼出し制限を維持する。
4. 設定例と README を更新し、MCP プロトコルを通る回帰テストで検証する。

## 実装

- `mcp>=1.24,<2` を直接依存に追加し、uv でロックファイルを更新。
- `kb_url` の既定値を `http://127.0.0.1:8000/mcp` に変更。
- 既存のホスト URL 設定は `/mcp` を補完し、パス付き設定はそのままエンドポイントとして使用。
- 認証ヘッダー・タイムアウトを設定した HTTP クライアントを MCP SDK に渡す。
- 構造化応答と旧形式の JSON テキスト応答を読み取り、MCP ツールエラーと KB のエラー応答を拒否。
- 文書のダウンロードには引き続き `encoding=text`、`include_images=false` を指定。
- kb-mcp のリポジトリおよび Git 管理対象外の実設定は変更しない。

## 検証

- MCP プロトコルのモックで初期化、初期化完了通知、ツール名・引数、セッション ID、終了を検査。
- 検索フィルター、重複排除、Markdown 限定、text/base64 原文と SHA-256 を確認。
- エラー応答、接続失敗からの再試行、タイムアウトを確認。
- 実 FastMCP をインメモリーで起動し、JSON と SSE 両形式で検索から原文取得まで確認。
- 全109件の pytest、Flake8（最大120文字）、`uv lock --check` が成功。
- 本番 KB（Qdrant・Embedding を含む）への実接続は別途必要。
