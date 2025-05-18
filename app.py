from flask import Flask, render_template, request, jsonify, session
from datetime import datetime, timedelta
import traceback # エラーログ出力用に保持
import os # 環境変数を読み込むため
import time
from dotenv import load_dotenv # 環境変数を読み込むため
import yfinance as yf
import pandas as pd
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from functools import wraps
from stock_analyzer import analyze_stocks
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

# CSRF保護の初期化
csrf = CSRFProtect()

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            token = request.headers.get('Authorization')
            if not token:
                return jsonify({'success': False, 'message': 'トークンが必要です'}), 401
            
            # Bearerトークンの形式を想定
            if token.startswith('Bearer '):
                token = token[7:]
            
            # トークンの検証
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            request.user = payload['user']
            return f(*args, **kwargs)
            
        except jwt.ExpiredSignatureError:
            return jsonify({'success': False, 'message': 'トークンの有効期限が切れています'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'success': False, 'message': '無効なトークンです'}), 401
        except Exception as e:
            return jsonify({'success': False, 'message': f'認証エラー: {str(e)}'}), 401
    return decorated_function

# 環境変数の読み込み
load_dotenv()

app = Flask(__name__)
# sessionを使うためにSECRET_KEYを設定
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config.update(
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=1),
    SQLALCHEMY_ENGINE_OPTIONS={'pool_pre_ping': True},
    WTF_CSRF_ENABLED=False  # CSRF保護を無効化
)

db = SQLAlchemy(app)

# ユーザーモデルの定義
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    portfolios = db.relationship('Portfolio', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'

# ポートフォリオモデルの定義
class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # 外部キーを追加
    symbol = db.Column(db.String(20), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    current_price = db.Column(db.Float, nullable=False)
    shares = db.Column(db.Integer, nullable=False)
    total_amount = db.Column(db.Float, nullable=False)
    expected_value = db.Column(db.Float, nullable=False)
    win_rate = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Portfolio {self.symbol}>'

# データベースの初期化
with app.app_context():
    try:
       
        # テーブルが存在しない場合のみ作成
        db.create_all()
        
    except Exception as e:
        print(f"データベースの初期化中にエラーが発生しました: {str(e)}")

def calculate_rsi(data, period=14):
    """RSIを計算する"""
    try:
        # 価格の差分を計算
        delta = data['Close'].diff()
        
        # 上昇と下落を分離
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 移動平均を計算
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # RSを計算（ゼロ除算を防ぐため、avg_lossが0の場合は小さな値を加える）
        rs = avg_gain / (avg_loss + 1e-10)
        
        # RSIを計算
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    except Exception as e:
        return pd.Series([None] * len(data), index=data.index)

def calculate_bollinger_bands(data, period=20, num_std=3):
    """ボリンジャーバンドを計算する"""
    try:
        if data is None or data.empty:
            return None, None, None, None

        # 移動平均を計算
        sma = data['Close'].rolling(window=period).mean()
        
        # 標準偏差を計算
        std = data['Close'].rolling(window=period).std()
        
        # バンドを計算
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        # 乖離率を計算
        deviation_upper = ((data['Close'] - upper_band) / upper_band) * 100
        deviation_lower = ((data['Close'] - lower_band) / lower_band) * 100
        
        return upper_band, lower_band, deviation_upper, deviation_lower
    except Exception as e:
        return None, None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    try:
        market = request.args.get('market', 'prime')  # デフォルトはプライム市場
        stock_list_file = 'growth_list.csv' if market == 'growth' else 'prime_list.csv'
        
        # 銘柄リストの読み込み
        df = pd.read_csv(stock_list_file)
        symbols = df['コード'].astype(str).tolist()
        
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

def normalize_ticker(ticker):
    """銘柄コードを正規化する"""
    try:
        if ticker is None:
            raise ValueError('銘柄コードが指定されていません')
            
        if not isinstance(ticker, str):
            raise ValueError('銘柄コードは文字列である必要があります')
            
        # 文字列の前後の空白文字（スペース、タブ、改行など）を削除
        # 英字は大文字に統一
        ticker = ticker.strip().upper()
        
        if not ticker:
            raise ValueError('銘柄コードが空です')
        
        # 日本株の処理
        # 4桁の数字、または4桁の数字+アルファベットの場合
        if len(ticker) == 4 and (ticker.isdigit() or (ticker[:-1].isdigit() and ticker[-1].isalpha())):
            return f"{ticker}.T"
        
        # その他の銘柄コードはそのまま返す
        return ticker
    except Exception as e:
        print(f"銘柄コードの正規化中にエラー: {str(e)}")
        raise ValueError(f"無効な銘柄コード: {str(e)}")

@app.route('/get_stock_list', methods=['POST'])
def get_stock_list():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'リクエストデータがありません'}), 400

        ticker = data.get('ticker')
        if not ticker:
            return jsonify({'error': '銘柄コードが指定されていません'}), 400

        try:
            period = int(data.get('period', 730))  # デフォルトは2年分
        except (TypeError, ValueError):
            return jsonify({'error': '期間の値が無効です'}), 400

        # 株価データの取得
        try:
            stock_data = get_stock_data(ticker, period)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            print(f"株価データ取得中にエラー: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': '株価データの取得に失敗しました'}), 500
        
        # データの整形（日付の降順にソート）
        formatted_data = []
        for item in stock_data:
            formatted_data.append({
                'date': item['date'],
                'open': item['open'],
                'high': item['high'],
                'low': item['low'],
                'close': item['close'],
                'volume': item['volume'],
                'rsi': item['rsi'],
                'lower_deviation': item['lower_deviation'],
                'short_term_expectation': item['short_term_expectation']
            })
        
        # 日付の降順にソート
        formatted_data.sort(key=lambda x: x['date'], reverse=True)

        # 銘柄名の取得
        stock_name = get_stock_name(ticker)

        return jsonify({
            'stock_name': stock_name,
            'stock_data': formatted_data
        })
    except Exception as e:
        print(f"エラー: {str(e)}")
        print(f"スタックトレース: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

def calculate_backtest(stock_data, buy_threshold=30, sell_threshold=0, disable_sell=False, shares=100):
    """バックテストを実行する"""
    try:
        trades = []
        positions = []  # 複数のポジションを管理するリスト
        latest_price = stock_data[0]['close']  # 最新の株価を保持
        latest_date = stock_data[0]['date']    # 最新の日付を保持

        # データを日付順にソート（新しい順）
        sorted_data = sorted(stock_data, 
                           key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d').replace(tzinfo=None),
                           reverse=True)
        
        # 最新のデータを取得
        latest_data = sorted_data[0]
        latest_price = latest_data['close']
        latest_date = latest_data['date']
        
        # バックテスト用にデータを古い順にソート
        sorted_data_oldest_first = sorted(stock_data,
                                        key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d').replace(tzinfo=None))
        
        # 古いデータから新しいデータへ処理
        for i in range(len(sorted_data_oldest_first)):
            current_data = sorted_data_oldest_first[i]
            current_date = datetime.strptime(current_data['date'], '%Y-%m-%d').replace(tzinfo=None)
            
            # 買いシグナルの確認
            if current_data['short_term_expectation'] is not None and \
               current_data['short_term_expectation'] >= buy_threshold:
                # 新しいポジションを追加
                position = {
                    'buy_date': current_data['date'],
                    'buy_price': current_data['close'],
                    'shares': shares,
                    'buy_expectation': current_data['short_term_expectation'],
                    'buy_date_obj': current_date
                }
                positions.append(position)
                
                # 売りなしの場合は即座に取引を記録
                if disable_sell:
                    trade = {
                        'buy_date': current_data['date'],
                        'sell_date': latest_date,
                        'buy_price': current_data['close'],
                        'sell_price': latest_price,
                        'shares': shares,
                        'buy_expectation': current_data['short_term_expectation'],
                        'sell_expectation': None,
                        'profit_rate': ((latest_price - current_data['close']) / current_data['close']) * 100,
                        'profit_amount': (latest_price - current_data['close']) * shares
                    }
                    trades.append(trade)
            
            # 売りシグナルの確認（売りなしでない場合のみ）
            elif not disable_sell and sell_threshold is not None and \
                 current_data['short_term_expectation'] is not None and \
                 current_data['short_term_expectation'] <= sell_threshold and \
                 len(positions) > 0:
                # 現在保有している全ポジションを処理
                for position in positions:
                    # 買付日より後の場合のみ売却
                    if current_date > position['buy_date_obj']:
                        trade = {
                            'buy_date': position['buy_date'],
                            'sell_date': current_data['date'],
                            'buy_price': position['buy_price'],
                            'sell_price': current_data['close'],
                            'shares': position['shares'],
                            'buy_expectation': position['buy_expectation'],
                            'sell_expectation': current_data['short_term_expectation'],
                            'profit_rate': ((current_data['close'] - position['buy_price']) / position['buy_price']) * 100,
                            'profit_amount': (current_data['close'] - position['buy_price']) * position['shares']
                        }
                        trades.append(trade)
                # ポジションをクリア
                positions = []

        # 最終ポジションの処理（売りなしでない場合のみ）
        if len(positions) > 0 and not disable_sell:
            latest_date_obj = datetime.strptime(latest_date, '%Y-%m-%d').replace(tzinfo=None)
            for position in positions:
                if latest_date_obj > position['buy_date_obj']:
                    trade = {
                        'buy_date': position['buy_date'],
                        'sell_date': latest_date,
                        'buy_price': position['buy_price'],
                        'sell_price': latest_price,
                        'shares': position['shares'],
                        'buy_expectation': position['buy_expectation'],
                        'sell_expectation': latest_data['short_term_expectation'],
                        'profit_rate': ((latest_price - position['buy_price']) / position['buy_price']) * 100,
                        'profit_amount': (latest_price - position['buy_price']) * position['shares']
                    }
                    trades.append(trade)
        
        # パフォーマンス指標の計算
        total_trades = len(trades)
        if total_trades > 0:
            winning_trades = len([t for t in trades if t['profit_rate'] > 0])
            losing_trades = total_trades - winning_trades
            total_profit_amount = sum(t['profit_amount'] for t in trades)
            average_profit_rate = sum(t['profit_rate'] for t in trades) / total_trades
            
            # 最大損失の計算（負の利益率の取引のみから計算）
            losing_trades_profits = [t['profit_rate'] for t in trades if t['profit_rate'] < 0]
            max_loss = min(losing_trades_profits) if losing_trades_profits else 0
            
            # 売却日ごとの利益率を平均化
            sell_date_profits = {}
            for trade in trades:
                sell_date = trade['sell_date']
                if sell_date not in sell_date_profits:
                    sell_date_profits[sell_date] = []
                sell_date_profits[sell_date].append(trade['profit_rate'])
            
            # 各売却日の平均利益率を計算
            avg_daily_profits = [sum(profits) / len(profits) for profits in sell_date_profits.values()]
            
            # 複利での総利益率の計算（売却日ごとの平均利益率を使用）
            # (1 + r1/100) * (1 + r2/100) * ... * (1 + rn/100) - 1) * 100
            total_profit_rate = (np.prod([1 + r/100 for r in avg_daily_profits]) - 1) * 100
            
            # 売りなしの場合は総利益率を平均利益率にする
            if disable_sell:
                total_profit_rate = average_profit_rate
        else:
            winning_trades = 0
            losing_trades = 0
            total_profit_amount = 0
            total_profit_rate = 0
            average_profit_rate = 0
            max_loss = 0
        
        return {
            'trades': trades,
            'performance': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                'total_profit_rate': total_profit_rate,
                'total_profit_amount': total_profit_amount,
                'average_profit_rate': average_profit_rate,
                'max_loss': max_loss
            }
        }
    except Exception as e:
        print(f"バックテスト中にエラーが発生しました: {str(e)}")
        print(traceback.format_exc())
        return None

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    try:
        data = request.get_json()
        ticker = data.get('ticker')
        buy_threshold = float(data.get('buy_threshold', 30))
        disable_sell = data.get('disable_sell', False)
        # 売りなしの場合はsell_thresholdをNoneに設定
        sell_threshold = None if disable_sell else float(data.get('sell_threshold', 0))
        period = int(data.get('period', 730))
        shares = int(data.get('shares', 100))  # デフォルトを100株に設定
        
        if not ticker:
            return jsonify({"error": "銘柄コードを入力してください"}), 400

        # 銘柄コードを正規化
        ticker_to_fetch = normalize_ticker(ticker)
        print(f"バックテストを実行する銘柄コード: {ticker_to_fetch}")

        # Yahoo Financeからデータを取得
        stock = yf.Ticker(ticker_to_fetch)
        
        # 銘柄名を取得
        try:
            stock_info = stock.info
            stock_name = stock_info.get('longName', '') or stock_info.get('shortName', '') or ticker_to_fetch
            if '.T' in ticker_to_fetch:  # 日本株の場合
                stock_name = f"{stock_name} ({ticker_to_fetch.replace('.T', '')})"
            else:
                stock_name = f"{stock_name} ({ticker_to_fetch})"
        except:
            stock_name = ticker_to_fetch
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period)  # 指定された期間分のデータを取得
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return jsonify({"error": f"銘柄コード '{ticker}' のデータが見つかりませんでした。"}), 404

        # RSIとボリンジャーバンドを計算
        rsi = calculate_rsi(hist)
        upper_band, lower_band, deviation_upper, deviation_lower = calculate_bollinger_bands(hist)

        # データを整形
        stock_data = []
        for date, row in hist.iterrows():
            try:
                # 日付をYYYY-MM-DD形式の文字列に変換
                formatted_date = date.strftime('%Y-%m-%d')

                # RSIと下乖離率の値を取得
                current_rsi = rsi.loc[date] if date in rsi.index else None
                current_lower_deviation = deviation_lower.loc[date] if date in deviation_lower.index else None

                # 数値に変換（NaNの場合はNone）
                rsi_value = round(float(current_rsi), 2) if not pd.isna(current_rsi) else None
                lower_deviation_value = round(float(current_lower_deviation), 2) if not pd.isna(current_lower_deviation) else None

                # 短期上昇期待値の計算
                short_term_expectation = None
                if rsi_value is not None and lower_deviation_value is not None:
                    rsi_component = 50 - rsi_value
                    deviation_component = abs(min(0, lower_deviation_value))
                    raw_expectation = rsi_component + deviation_component
                    # 0以上の場合は値*2にする
                    short_term_expectation = round(raw_expectation * 2, 2) if raw_expectation >= 0 else round(raw_expectation, 2)

                stock_data.append({
                    "date": formatted_date,
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": int(row['Volume']),
                    "rsi": rsi_value,
                    "lower_deviation": lower_deviation_value,
                    "short_term_expectation": short_term_expectation
                })
            except (ValueError, TypeError, KeyError) as e:
                print(f"データ処理中にエラー: Date={date}, Data={row}, エラー: {str(e)}")
                continue

        # バックテストを実行
        backtest_result = calculate_backtest(stock_data, buy_threshold, sell_threshold, disable_sell, shares)
        
        if backtest_result is None:
            return jsonify({"error": "バックテストの実行中にエラーが発生しました"}), 500

        # 銘柄名を結果に追加
        backtest_result['stock_name'] = stock_name

        # stock_data も含めて返すように修正
        response_data = {
            **backtest_result,  # trades と performance を展開して追加
            'stock_data': stock_data # 整形済みの株価データを追加
        }

        return jsonify(response_data)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": f"バックテスト中にエラーが発生しました: {str(e)}"}), 500

def calculate_sdi(data, period=20):
    """SDI（Standard Deviation Index）を計算する"""
    try:
        # 移動平均を計算
        sma = data['Close'].rolling(window=period).mean()
        # 標準偏差を計算
        std = data['Close'].rolling(window=period).std()
        # SDIを計算（標準偏差を移動平均で割って1000を掛け、スケーリング）
        sdi = (std / sma) * 1000
        # 最小値と最大値を設定
        min_sdi = 0
        max_sdi = 100
        # スケーリングを行う（0〜100の範囲に変換）
        sdi = ((sdi - sdi.min()) / (sdi.max() - sdi.min())) * (max_sdi - min_sdi) + min_sdi
        # 0〜100の範囲に収める
        sdi = sdi.clip(0, 100)
        return sdi
    except Exception as e:
        return pd.Series([None] * len(data), index=data.index)

def get_latest_stock_info(ticker_to_fetch):
    """指定された銘柄の最新情報を取得するヘルパー関数"""
    try:
        if not ticker_to_fetch:
            raise ValueError("銘柄コードが指定されていません")
            
        print(f"銘柄情報の取得を開始: {ticker_to_fetch}")
        
        # 銘柄コードの正規化
        if isinstance(ticker_to_fetch, str):
            ticker_to_fetch = ticker_to_fetch.strip()
            if not ticker_to_fetch.endswith('.T') and len(ticker_to_fetch) == 4 and ticker_to_fetch.isdigit():
                ticker_to_fetch = f"{ticker_to_fetch}.T"
        else:
            raise ValueError(f"無効な銘柄コード形式です: {ticker_to_fetch}")
            
        print(f"正規化後の銘柄コード: {ticker_to_fetch}")
        
        stock = yf.Ticker(ticker_to_fetch)

        # 銘柄名を取得
        try:
            print(f"銘柄の基本情報を取得中: {ticker_to_fetch}")
            stock_info = stock.info
            if not stock_info:
                raise ValueError("銘柄情報が取得できませんでした")
                
            stock_name = stock_info.get('longName', '') or stock_info.get('shortName', '') or ticker_to_fetch
            print(f"取得した銘柄名: {stock_name}")
            
            if '.T' in ticker_to_fetch:  # 日本株の場合
                display_ticker = ticker_to_fetch.replace('.T', '')
                stock_name = f"{stock_name} ({display_ticker})"
            else:
                stock_name = f"{stock_name} ({ticker_to_fetch})"
        except Exception as e:
            print(f"銘柄名の取得に失敗: {str(e)}")
            error_details = str(e)
            stock_name = f"{ticker_to_fetch} (エラー: {error_details})"

        # 株価データを取得
        print(f"株価データの取得を開始: {ticker_to_fetch}")
        hist = stock.history(period="3mo")

        if hist.empty:
            error_msg = f"銘柄コード '{ticker_to_fetch}' のデータが見つかりませんでした。"
            print(error_msg)
            return {
                "ticker": ticker_to_fetch.replace('.T', '') if '.T' in ticker_to_fetch else ticker_to_fetch,
                "name": stock_name,
                "latest_price": None,
                "short_term_expectation": None,
                "error": error_msg
            }

        # 最新の価格を取得
        latest_price = hist['Close'].iloc[-1]
        print(f"最新の株価: {latest_price}")

        # 最新の指標を計算
        print(f"テクニカル指標の計算を開始: {ticker_to_fetch}")
        rsi = calculate_rsi(hist)
        upper_band, lower_band, deviation_upper, deviation_lower = calculate_bollinger_bands(hist)

        latest_rsi = rsi.iloc[-1] if not rsi.empty and len(rsi) > 0 else None
        latest_lower_deviation = deviation_lower.iloc[-1] if not deviation_lower.empty and len(deviation_lower) > 0 else None

        print(f"計算結果 - RSI: {latest_rsi}, 下方乖離率: {latest_lower_deviation}")

        # 最新の短期上昇期待値を計算
        short_term_expectation = None
        if latest_rsi is not None and latest_lower_deviation is not None:
            rsi_component = 50 - latest_rsi
            deviation_component = abs(min(0, latest_lower_deviation))
            raw_expectation = rsi_component + deviation_component
            short_term_expectation = round(raw_expectation * 2, 2) if raw_expectation >= 0 else round(raw_expectation, 2)
            print(f"短期上昇期待値: {short_term_expectation}")

        return {
            "ticker": ticker_to_fetch.replace('.T', '') if '.T' in ticker_to_fetch else ticker_to_fetch,
            "name": stock_name,
            "latest_price": round(float(latest_price), 2) if latest_price is not None else None,
            "short_term_expectation": short_term_expectation
        }

    except Exception as e:
        error_message = f"銘柄情報取得中にエラー発生 ({ticker_to_fetch}): {str(e)}"
        print(error_message)
        print(traceback.format_exc())
        return {
            "ticker": ticker_to_fetch.replace('.T', '') if '.T' in ticker_to_fetch else ticker_to_fetch,
            "name": f"{ticker_to_fetch} (エラー: {str(e)})",
            "latest_price": None,
            "short_term_expectation": None,
            "error": error_message
        }

@app.route('/add_to_portfolio', methods=['POST'])
def add_to_portfolio():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'リクエストデータがありません'
            }), 400

        ticker = data.get('ticker')
        token = data.get('token')

        if not ticker or not token:
            return jsonify({
                'success': False,
                'message': '必要なデータが不足しています'
            }), 400

        try:
            # トークンの検証
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            username = payload['user']
            
            # ユーザーの取得
            user = User.query.filter_by(username=username).first()
            if not user:
                return jsonify({
                    'success': False,
                    'message': 'ユーザーが見つかりません'
                }), 404

            # 銘柄コードの正規化
            normalized_ticker = normalize_ticker(ticker)

            # 既にポートフォリオに存在するかチェック
            existing = Portfolio.query.filter_by(user_id=user.id, ticker=normalized_ticker).first()
            if existing:
                return jsonify({
                    'success': False,
                    'message': 'この銘柄は既にポートフォリオに存在します'
                }), 400

            # ポートフォリオに追加
            new_portfolio = Portfolio(
                user_id=user.id,
                ticker=normalized_ticker
            )
            db.session.add(new_portfolio)
            db.session.commit()

            return jsonify({
                'success': True,
                'message': 'ポートフォリオに追加しました'
            })

        except jwt.ExpiredSignatureError:
            return jsonify({
                'success': False,
                'message': 'トークンの有効期限が切れています'
            }), 401
        except jwt.InvalidTokenError:
            return jsonify({
                'success': False,
                'message': '無効なトークンです'
            }), 401

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'エラーが発生しました: {str(e)}'
        }), 500

@app.route('/remove_from_portfolio', methods=['POST'])
def remove_from_portfolio():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'リクエストデータがありません'
            }), 400

        ticker = data.get('ticker')
        token = data.get('token')

        if not ticker or not token:
            return jsonify({
                'success': False,
                'message': '必要なデータが不足しています'
            }), 400

        try:
            # トークンの検証
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            username = payload['user']
            
            # ユーザーの取得
            user = User.query.filter_by(username=username).first()
            if not user:
                return jsonify({
                    'success': False,
                    'message': 'ユーザーが見つかりません'
                }), 404

            # ポートフォリオから削除
            portfolio = Portfolio.query.filter_by(user_id=user.id, ticker=ticker).first()
            if portfolio:
                db.session.delete(portfolio)
                db.session.commit()

            return jsonify({
                'success': True,
                'message': 'ポートフォリオから削除しました'
            })

        except jwt.ExpiredSignatureError:
            return jsonify({
                'success': False,
                'message': 'トークンの有効期限が切れています'
            }), 401
        except jwt.InvalidTokenError:
            return jsonify({
                'success': False,
                'message': '無効なトークンです'
            }), 401

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'エラーが発生しました: {str(e)}'
        }), 500

@app.route('/get_portfolio', methods=['GET'])
def get_portfolio():
    try:
        portfolios = Portfolio.query.all()
        portfolio_list = []
        total_investment = 0
        total_value = 0
        total_profit = 0
        total_profit_rate = 0

        for portfolio in portfolios:
            # 最新の株価情報を取得
            stock_info = get_latest_stock_info(portfolio.symbol)
            if stock_info and 'error' not in stock_info:
                current_price = stock_info['latest_price']
                if current_price is not None:
                    # 現在の評価額と損益を計算
                    current_value = current_price * portfolio.shares
                    profit_amount = current_value - portfolio.total_amount
                    profit_rate = (profit_amount / portfolio.total_amount) * 100

                    total_investment += portfolio.total_amount
                    total_value += current_value
                    total_profit += profit_amount
                    total_profit_rate += profit_rate

                    portfolio_list.append({
                        'symbol': portfolio.symbol,
                        'name': portfolio.name,
                        'current_price': current_price,
                        'shares': portfolio.shares,
                        'total_amount': portfolio.total_amount,
                        'current_value': current_value,
                        'profit_rate': profit_rate,
                        'profit_amount': profit_amount,
                        'expected_value': portfolio.expected_value,
                        'win_rate': portfolio.win_rate
                    })

        avg_profit_rate = total_profit_rate / len(portfolios) if portfolios else 0
        avg_profit_amount = total_profit / len(portfolios) if portfolios else 0
        total_profit_rate_value = (total_profit / total_investment) * 100 if total_investment > 0 else 0

        return jsonify({
            'success': True,
            'portfolio_list': portfolio_list,
            'summary': {
                'total_investment': total_investment,
                'total_value': total_value,
                'total_profit': total_profit,
                'total_profit_rate': total_profit_rate_value,
                'avg_profit_rate': avg_profit_rate,
                'avg_profit_amount': avg_profit_amount
            }
        })

    except Exception as e:
        print(f"ポートフォリオ取得中にエラー: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'エラーが発生しました: {str(e)}'
        }), 500

@app.route('/update_portfolio', methods=['POST'])
def update_portfolio():
    try:
        data = request.get_json()
        if not data or 'portfolio_list' not in data:
            return jsonify({
                'success': False,
                'message': 'リクエストデータが不正です'
            }), 400

        # 既存のポートフォリオを全て削除
        Portfolio.query.delete()

        # 新しいポートフォリオを追加
        for item in data['portfolio_list']:
            portfolio = Portfolio(
                symbol=item['symbol'],
                name=item['name'],
                current_price=item['current_price'],
                shares=item['shares'],
                total_amount=item['total_amount'],
                expected_value=item['expected_value'],
                win_rate=item['win_rate']
            )
            db.session.add(portfolio)

        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'ポートフォリオを更新しました'
        })

    except Exception as e:
        db.session.rollback()
        print(f"ポートフォリオ更新中にエラー: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'エラーが発生しました: {str(e)}'
        }), 500

def get_stock_data(ticker, period):
    """株価データを取得する"""
    try:
        ticker_to_fetch = normalize_ticker(ticker)
        stock = yf.Ticker(ticker_to_fetch)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            raise Exception(f"銘柄コード '{ticker}' のデータが見つかりませんでした。")
        
        # RSIとボリンジャーバンドを計算
        rsi = calculate_rsi(hist)
        upper_band, lower_band, deviation_upper, deviation_lower = calculate_bollinger_bands(hist)
        
        stock_data = []
        for date, row in hist.iterrows():
            try:
                current_rsi = rsi.loc[date] if date in rsi.index else None
                current_lower_deviation = deviation_lower.loc[date] if date in deviation_lower.index else None
                
                rsi_value = round(float(current_rsi), 2) if not pd.isna(current_rsi) else None
                lower_deviation_value = round(float(current_lower_deviation), 2) if not pd.isna(current_lower_deviation) else None
                
                short_term_expectation = None
                if rsi_value is not None and lower_deviation_value is not None:
                    rsi_component = 50 - rsi_value
                    deviation_component = abs(min(0, lower_deviation_value))
                    raw_expectation = rsi_component + deviation_component
                    short_term_expectation = round(raw_expectation * 2, 2) if raw_expectation >= 0 else round(raw_expectation, 2)
                
                date_str = date.strftime('%Y-%m-%d')
                
                stock_data.append({
                    'date': date_str,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']),
                    'rsi': rsi_value,
                    'lower_deviation': lower_deviation_value,
                    'short_term_expectation': short_term_expectation
                })
            except (ValueError, TypeError, KeyError) as e:
                print(f"データ処理中にエラー: Date={date}, Data={row}, エラー: {str(e)}")
                continue
        
        if not stock_data:
            raise Exception(f"銘柄コード '{ticker}' の有効なデータが見つかりませんでした。")
            
        return stock_data
        
    except Exception as e:
        print(f"株価データ取得中にエラー: {str(e)}")
        raise Exception(f"銘柄コード '{ticker}' のデータ取得に失敗しました: {str(e)}")

def get_stock_name(ticker):
    """銘柄名を取得する"""
    ticker_to_fetch = normalize_ticker(ticker)
    stock = yf.Ticker(ticker_to_fetch)
    try:
        stock_info = stock.info
        stock_name = stock_info.get('longName', '') or stock_info.get('shortName', '') or ticker_to_fetch
        if '.T' in ticker_to_fetch:
            stock_name = f"{stock_name} ({ticker_to_fetch.replace('.T', '')})"
        else:
            stock_name = f"{stock_name} ({ticker_to_fetch})"
    except:
        stock_name = ticker_to_fetch
    return stock_name

def analyze_single_stock(symbol):
    """単一銘柄の分析を行う"""
    try:
        # 銘柄コードの正規化
        ticker = normalize_ticker(symbol)
        
        # 株価データの取得（期間を1年に短縮）
        stock_data = get_stock_data(ticker, 365)  # 1年分のデータを取得
        
        if not stock_data:
            return None
            
        # 最新のデータを取得
        latest_data = stock_data[0]
        
        # 移動平均の計算（最新の20日分のみを使用）
        closes = [d['close'] for d in stock_data[:20]]
        ma20 = sum(closes) / len(closes) if len(closes) == 20 else None
        
        # 期待値の計算（最新の30日分のみを使用）
        expected_values = [d['short_term_expectation'] for d in stock_data[:30] if d['short_term_expectation'] is not None]
        if not expected_values:
            return None
            
        current_expected_value = expected_values[0]
        max_expected_value = max(expected_values)
        min_expected_value = min(expected_values)
        
        # バックテストの実行（期間を短縮）
        backtest_result = calculate_backtest(stock_data[:180])  # 6ヶ月分のデータでバックテスト
        
        # バックテスト結果の取得
        backtest_performance = None
        if backtest_result and 'performance' in backtest_result:
            backtest_performance = {
                'win_rate': backtest_result['performance'].get('win_rate', 0),
                'avg_profit': backtest_result['performance'].get('average_profit_rate', 0),
                'max_profit': backtest_result['performance'].get('total_profit_rate', 0),
                'max_loss': backtest_result['performance'].get('max_loss', 0),
                'total_trades': backtest_result['performance'].get('total_trades', 0)
            }
        
        # 結果の作成
        result = {
            'symbol': symbol,
            'name': get_stock_name(symbol),
            'current_price': latest_data['close'],
            'ma20': ma20 if ma20 is not None else latest_data['close'],
            'expected_value': current_expected_value,
            'max_expected_value': max_expected_value,
            'min_expected_value': min_expected_value,
            'backtest': backtest_performance
        }
        
        return result
        
    except Exception as e:
        return None

if __name__ == '__main__':
    print("Flaskアプリケーションを起動中...")
    print("データベースを初期化中...")
    with app.app_context():
        db.create_all()
    print("データベースの初期化が完了しました")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 