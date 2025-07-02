#!/bin/bash

# アプリケーションのディレクトリに移動
cd "/Users/a0000/上昇期待値指数"

# Pythonの仮想環境をアクティベート（もし存在する場合）
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 必要なパッケージをインストール
pip install -r requirements.txt

# ポート5002で起動
export PORT=5002

# バックグラウンドでブラウザを開く
(sleep 3 && open http://localhost:5002) &

# アプリケーションを起動
python app.py 