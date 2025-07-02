import os
import sys
import subprocess
import webbrowser
from time import sleep

def run_app():
    # 必要なパッケージをインストール
    required_packages = [
        'flask',
        'flask-sqlalchemy',
        'flask-wtf',
        'python-dotenv',
        'yfinance',
        'pandas',
        'numpy',
        'pyjwt',
        'werkzeug'
    ]
    
    print("必要なパッケージをインストール中...")
    for package in required_packages:
        subprocess.run([sys.executable, "-m", "pip", "install", package])
    
    # アプリケーションを起動
    print("アプリケーションを起動中...")
    app_process = subprocess.Popen([sys.executable, "app.py"])
    
    # ブラウザを開く
    sleep(2)  # サーバーが起動するまで少し待機
    webbrowser.open('http://localhost:5000')
    
    try:
        app_process.wait()
    except KeyboardInterrupt:
        app_process.terminate()
        print("\nアプリケーションを終了しました。")

if __name__ == "__main__":
    run_app() 