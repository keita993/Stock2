from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import traceback
import os
import time
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from stock_analyzer import (
    analyze_stocks,
    analyze_single_stock,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_expected_value,
    get_stock_data
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import pytz  # タイムゾーン用に追加

# CSRF保護の初期化
csrf = CSRFProtect()

# 環境変数の読み込み
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///trades.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config.update(
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=1),
    WTF_CSRF_ENABLED=False
)

db = SQLAlchemy(app)

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    buy_price = db.Column(db.Float, nullable=False)
    buy_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    sell_price = db.Column(db.Float)
    sell_date = db.Column(db.DateTime)
    status = db.Column(db.String(10), nullable=False, default='open')  # 'open' or 'closed'
    profit_loss = db.Column(db.Float)
    quantity = db.Column(db.Integer, nullable=False, default=100)  # 100株単位

# データベースの初期化
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    try:
        market = request.args.get('market', 'prime')  # デフォルトはプライム市場
        print(f"選択された市場: {market}")  # デバッグ用
        
        # 市場に応じて銘柄リストファイルを選択
        if market == 'growth':
            stock_list_file = 'growth_list.csv'
        else:
            stock_list_file = 'prime_list.csv'
        
        print(f"使用する銘柄リスト: {stock_list_file}")  # デバッグ用
        
        # 銘柄リストの読み込み
        try:
            df = pd.read_csv(stock_list_file)
            symbols = df['コード'].astype(str).tolist()
            print(f"読み込んだ銘柄数: {len(symbols)}")  # デバッグ用
        except Exception as e:
            print(f"銘柄リストの読み込みエラー: {str(e)}")  # デバッグ用
            return jsonify({'error': f'銘柄リストの読み込みに失敗しました: {str(e)}'})
        
        # 分析の実行（並列処理）
        results = []
        total_stocks = len(symbols)
        analyzed_stocks = 0
        
        # CPUコア数を取得（-1で全コア使用）
        max_workers = multiprocessing.cpu_count() * 8  # コア数の8倍のワーカーを使用
        
        print(f"\n{market}市場の分析を開始します...")
        
        # バッチサイズを設定（一度に処理する銘柄数）
        batch_size = 200  # バッチサイズを増加
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 進捗バーの初期化
            pbar = tqdm(total=total_stocks, desc="分析進捗", unit="銘柄", ncols=80)
            
            # バッチ処理
            for i in range(0, total_stocks, batch_size):
                batch_symbols = symbols[i:i + batch_size]
                
                # 各銘柄の分析を並列実行
                future_to_symbol = {
                    executor.submit(analyze_single_stock, symbol): symbol 
                    for symbol in batch_symbols
                }
                
                # 完了した分析結果を収集
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result and result['expected_value'] >= 30:
                            results.append(result)
                            analyzed_stocks += 1
                    except Exception:
                        pass
                    finally:
                        pbar.update(1)
            
            pbar.close()
        
        print(f"\n分析完了: {analyzed_stocks}件の銘柄が見つかりました")
        
        return jsonify({
            'results': results,
            'total_stocks': total_stocks,
            'analyzed_stocks': analyzed_stocks
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_stock_list', methods=['POST'])
def get_stock_list():
    try:
        data = request.get_json()
        symbol = data.get('ticker')
        
        # 日本株の場合は.Tを付加
        if not symbol.endswith('.T'):
            symbol = f"{symbol}.T"
        
        # 株価データの取得（2年分）
        stock = yf.Ticker(symbol)
        hist = stock.history(period='2y')
        
        if hist.empty:
            return jsonify({'error': 'データが見つかりません'})
        
        # 銘柄名の取得
        stock_name = stock.info.get('longName', '') or stock.info.get('shortName', '') or symbol
        
        # RSIを計算
        rsi = calculate_rsi(hist)
        if rsi is None:
            return jsonify({'error': 'RSIの計算に失敗しました'})
        
        # ボリンジャーバンドを計算
        upper_band, lower_band, deviation_upper, deviation_lower = calculate_bollinger_bands(hist)
        if lower_band is None:
            return jsonify({'error': 'ボリンジャーバンドの計算に失敗しました'})
        
        # データの整形
        stock_data = []
        for i in range(len(hist)):
            if i < 20:  # 20日分のデータはスキップ
                continue
                
            current_rsi = rsi.iloc[i]
            current_lower_deviation = deviation_lower.iloc[i]
            
            # RSIコンポーネント
            rsi_component = 50 - current_rsi
            
            # 下方乖離率コンポーネント
            deviation_component = abs(min(0, current_lower_deviation))
            
            # 期待値の計算
            raw_expectation = rsi_component + deviation_component
            expected_value = round(raw_expectation * 2, 2) if raw_expectation >= 0 else round(raw_expectation, 2)
            
            stock_data.append({
                'date': hist.index[i].strftime('%Y-%m-%d'),
                'close': hist['Close'].iloc[i],
                'short_term_expectation': expected_value
            })
        
        return jsonify({
            'stock_name': stock_name,
            'stock_data': stock_data
        })
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/buy_stock', methods=['POST'])
def buy_stock():
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        name = data.get('name')
        price = float(data.get('price'))
        quantity = int(data.get('quantity', 100))  # デフォルトは100株
        
        # 株数が100の倍数であることを確認
        if quantity % 100 != 0:
            return jsonify({'error': '株数は100株単位で指定してください'}), 400
        
        # 新しい取引を作成（UTCで保存）
        trade = Trade(
            symbol=symbol,
            name=name,
            buy_price=price,
            quantity=quantity
        )
        
        db.session.add(trade)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': '購入が完了しました',
            'trade_id': trade.id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/sell_stock', methods=['POST'])
def sell_stock():
    try:
        data = request.get_json()
        trade_id = data.get('trade_id')
        
        # 取引を検索
        trade = Trade.query.get(trade_id)
        if not trade:
            return jsonify({'error': '取引が見つかりません'})
            
        if trade.status == 'closed':
            return jsonify({'error': 'この取引は既に終了しています'})
        
        # 最新の株価を取得
        symbol = trade.symbol
        if not symbol.endswith('.T'):
            symbol = f"{symbol}.T"
        
        stock = yf.Ticker(symbol)
        hist = stock.history(period='1d')
        
        if hist.empty:
            return jsonify({'error': '最新の株価を取得できませんでした'})
        
        sell_price = hist['Close'].iloc[-1]
        
        # 取引を更新（UTCで保存）
        trade.sell_price = sell_price
        trade.sell_date = datetime.utcnow()
        trade.status = 'closed'
        trade.profit_loss = (sell_price - trade.buy_price) * trade.quantity
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': '売却が完了しました',
            'profit_loss': trade.profit_loss,
            'sell_price': sell_price
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_trades', methods=['GET'])
def get_trades():
    try:
        status = request.args.get('status', 'all')  # 'all', 'open', 'closed'
        
        query = Trade.query
        if status != 'all':
            query = query.filter_by(status=status)
            
        trades = query.order_by(Trade.buy_date.desc()).all()
        
        # 東京時間に変換
        jst = pytz.timezone('Asia/Tokyo')
        
        return jsonify({
            'trades': [{
                'id': trade.id,
                'symbol': trade.symbol,
                'name': trade.name,
                'buy_price': trade.buy_price,
                'buy_date': (trade.buy_date + timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'),
                'sell_price': trade.sell_price,
                'sell_date': (trade.sell_date + timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S') if trade.sell_date else None,
                'status': trade.status,
                'profit_loss': trade.profit_loss,
                'quantity': trade.quantity
            } for trade in trades]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_stock_price')
def get_stock_price():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': '銘柄コードが指定されていません'}), 400
    
    try:
        # 銘柄コードから.Tを除去（既に付いている場合）
        symbol = symbol.replace('.T', '')
        
        # yfinanceを使用して最新の株価を取得
        stock = yf.Ticker(f"{symbol}.T")
        hist = stock.history(period="1d")
        
        if hist.empty:
            print(f"警告: {symbol}の株価データが空です")
            return jsonify({'error': f'{symbol}の株価データを取得できませんでした'}), 404
        
        current_price = hist['Close'].iloc[-1]
        print(f"成功: {symbol}の現在値 {current_price}を取得")
        return jsonify({
            'symbol': symbol,
            'current_price': current_price
        })
    except Exception as e:
        error_msg = f'株価取得中にエラーが発生しました: {str(e)}'
        print(f"エラー: {error_msg}")
        print(f"詳細なエラー情報: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@app.route('/get_stock_expectation')
def get_stock_expectation():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': '銘柄コードが指定されていません'})
    
    try:
        # 日本株の場合は.Tを付加
        if not symbol.endswith('.T'):
            symbol = f"{symbol}.T"
        
        print(f"期待値計算開始: {symbol}")  # デバッグログ
        
        # 株価データを取得
        hist_data = get_stock_data(symbol)
        if hist_data is None or len(hist_data) < 20:
            print(f"株価データの取得に失敗: {symbol}")  # デバッグログ
            return jsonify({'error': '株価データの取得に失敗しました'})
        
        print(f"取得したデータ期間: {hist_data.index[0]} から {hist_data.index[-1]}")  # デバッグログ
        print(f"データ件数: {len(hist_data)}")  # デバッグログ
        
        # 期待値を計算
        expected_value, max_expected_value, min_expected_value = calculate_expected_value(hist_data)
        if expected_value is None:
            print(f"期待値の計算に失敗: {symbol}")  # デバッグログ
            return jsonify({'error': '期待値の計算に失敗しました'})
        
        print(f"計算された期待値: {expected_value}")  # デバッグログ
        
        return jsonify({
            'expected_value': expected_value,
            'max_expected_value': max_expected_value,
            'min_expected_value': min_expected_value
        })
        
    except Exception as e:
        print(f"期待値計算中にエラーが発生: {str(e)}")  # デバッグログ
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Flaskアプリケーションを起動中...")
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True) 