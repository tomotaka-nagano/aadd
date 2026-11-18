# exmgai 0.0.10 更新

## 計画

- ローカル editable 参照を維持し、必要バージョンを 0.0.10 以上に更新する。
- uv のロックと依存環境を更新する。
- タスク停止後に生成・判断用クライアントを閉じ、ライブラリ側の資格情報解放を実行する。
- 終了順序、複数クライアント、一部の解放失敗、ライフサイクル例外時の解放を検証する。

## 実装

- AI.close でキャッシュした全クライアントの close を実行する。
  一部が失敗しても他を解放し、失敗は例外グループとして呼出し元に通知する。
- Workflow.close は実行タスクのキャンセル・終了を待ってから AI.close を呼ぶ。
- FastAPI の lifespan は finally で Workflow.close を実行する。
- テスト用 AI も同じ終了インターフェースに合わせる。

## 検証

- `uv sync`: 成功。ローカル editable 参照の exmgai 0.0.9 を 0.0.10 に更新した。
- 実行環境のメタデータと import 先で、0.0.10 とローカル参照先を確認した。
- `uv run python -m pytest -q`: アプリの全93件成功。
- `uv run python -m pytest -q ..\exmgai\tests\test_auth.py ..\exmgai\tests\test_client.py ..\exmgai\tests\test_providers.py`:
  ライブラリの認証・クライアント・プロバイダーの34件成功。
  ライブラリ側設定に起因する pytest-asyncio の fixture loop scope 未指定の非推奨警告はあるが、失敗はない。
- `uv run python -m flake8 src tests --max-line-length=120`: 成功。
- `uv lock --check` / `git diff --check`: 成功。
- Pylance: 修正した AI・Workflow・ライフサイクルに診断エラーなし。
- テストはモック認証・HTTP を使用し、実 Azure サービスへの接続は検証していない。
