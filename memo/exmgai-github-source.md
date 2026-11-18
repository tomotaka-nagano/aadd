# exmgai の GitHub 参照への変更

## 計画

- exmgai のローカル editable 参照を、GitHub のリリースタグ v0.0.10 への参照へ変更する。
- uv でロックと依存環境を更新し、取得元と回帰テストを確認する。

## 実装

- 依存条件 exmgai>=0.0.10 を維持し、tool.uv.sources に Git URL と v0.0.10 タグを指定した。
- README のローカル配置前提を削除し、非公開 Git リポジトリへの認証と更新手順を記載した。

## 検証

- GitHub 上の v0.0.10 タグが存在することを確認した。
- `uv sync`: 成功。ローカル editable 版を GitHub 取得版へ置き換えた。
- インストール済みパッケージのメタデータでバージョン 0.0.10、GitHub URL、要求タグ v0.0.10、
  コミット e0bbe356f673978eb1cad470b1b3b074b141b24f を確認した。
- `uv run python -m pytest -q`: 全93件成功。
- `uv lock --check` / `git diff --check`: 成功。
