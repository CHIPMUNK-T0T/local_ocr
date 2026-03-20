# Local OCR Project

このプロジェクトは、Pythonライブラリ `docling` を使用してOCR処理を行うための環境です。

## セットアップ情報

- **仮想環境名:** `ocr_venv`
- **Pythonバージョン:** 3.12.3
- **主要ライブラリ:** `docling`

## 仮想環境の使用方法

以下のコマンドで仮想環境を有効化できます。

```bash
source ocr_venv/bin/activate
```

## 開発ルール
- 依存関係を追加した場合は、`pip freeze > requirements.txt` を実行して更新してください。
- 実装の詳細は `.gemini/GEMINI.md`（存在する場合）の指示に従ってください。
- モデルごとの依存競合を避けるため、OCR モデルは仮想環境を分離して管理します。
- 現在動いている環境はそのまま活用し、動いていないモデルだけを新しい仮想環境へ移行します。
- 各モデルクラスは、自分に割り当てられた仮想環境の Python を使って worker を呼び出せるようにします。

## 仮想環境ポリシー

- `docling`: 現在は `ocr_venv`
- `glm`: 現在は `ocr_venv`
- `dots`: `venvs/dots_vllm_venv`
- `paddle`: `venvs/paddle_venv`
