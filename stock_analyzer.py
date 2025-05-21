import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures
from tqdm import tqdm

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
        return None

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

def get_stock_data(symbol, period='2y'):
    """
    指定された銘柄の株価データを取得（2年分）
    """
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        return None

def calculate_expected_value(hist_data):
    """
    株価期待値を計算（RSIとボリンジャーバンドの複合計算）
    """
    if hist_data is None or len(hist_data) < 20:
        return None, None, None
    
    try:
        # RSIを計算
        rsi = calculate_rsi(hist_data)
        if rsi is None:
            return None, None, None
        
        # ボリンジャーバンドを計算
        upper_band, lower_band, deviation_upper, deviation_lower = calculate_bollinger_bands(hist_data)
        if lower_band is None:
            return None, None, None
        
        # 全期間の期待値を計算
        expected_values = []
        for i in range(len(hist_data)):
            if i < 20:  # 20日分のデータが必要
                continue
                
            current_rsi = rsi.iloc[i]
            current_lower_deviation = deviation_lower.iloc[i]
            
            # RSIコンポーネント
            rsi_component = 50 - current_rsi
            
            # 下方乖離率コンポーネント
            deviation_component = abs(min(0, current_lower_deviation))
            
            # 期待値の計算
            raw_expectation = rsi_component + deviation_component
            
            # 期待値の最終調整
            expected_value = round(raw_expectation * 2, 2) if raw_expectation >= 0 else round(raw_expectation, 2)
            expected_values.append(expected_value)
        
        if not expected_values:
            return None, None, None
            
        # 最新の期待値、最大値、最小値を返す
        current_expected_value = expected_values[-1]
        max_expected_value = max(expected_values)
        min_expected_value = min(expected_values)
        
        return current_expected_value, max_expected_value, min_expected_value
        
    except Exception as e:
        return None, None, None

def get_stock_name(symbol):
    """
    銘柄名を取得する
    """
    try:
        stock = yf.Ticker(symbol)
        stock_info = stock.info
        stock_name = stock_info.get('longName', '') or stock_info.get('shortName', '') or symbol
        if '.T' in symbol:  # 日本株の場合
            stock_name = f"{stock_name} ({symbol.replace('.T', '')})"
        return stock_name
    except Exception as e:
        print(f"銘柄名の取得に失敗: {symbol}, エラー: {e}")
        return symbol

def calculate_backtest(hist_data, expected_values, buy_threshold=30, sell_threshold=0):
    """
    バックテストを実行し、勝率を計算
    期待値が30以上で買い続け、0以下で一括売却
    """
    if hist_data is None or len(hist_data) < 20 or len(expected_values) < 20:
        return None
        
    try:
        trades = []
        positions = []  # 複数のポジションを管理
        
        # データを古い順にソート
        dates = hist_data.index[20:]  # 20日目以降のデータを使用
        prices = hist_data['Close'].values[20:]
        expectations = expected_values[20:]  # 20日目以降の期待値を使用
        
        for i in range(len(dates)):
            if i >= len(expectations):  # インデックスチェック
                break
                
            current_price = prices[i]
            current_expectation = expectations[i]
            
            # NaNチェック
            if pd.isna(current_price) or pd.isna(current_expectation):
                continue
            
            # 買いシグナル（期待値が30以上）
            if current_expectation >= buy_threshold:
                positions.append({
                    'buy_date': dates[i].strftime('%Y-%m-%d'),
                    'buy_price': float(current_price),
                    'buy_expectation': float(current_expectation)
                })
            
            # 売りシグナル（期待値が0以下）
            elif current_expectation <= sell_threshold and positions:
                # 全ポジションを一括売却
                for position in positions:
                    profit_rate = ((current_price - position['buy_price']) / position['buy_price']) * 100
                    trades.append({
                        'buy_date': position['buy_date'],
                        'sell_date': dates[i].strftime('%Y-%m-%d'),
                        'buy_price': float(position['buy_price']),
                        'sell_price': float(current_price),
                        'profit_rate': float(profit_rate),
                        'buy_expectation': float(position['buy_expectation']),
                        'sell_expectation': float(current_expectation)
                    })
                positions = []  # ポジションをクリア
        
        # 最終ポジションの処理
        if positions:
            for position in positions:
                profit_rate = ((prices[-1] - position['buy_price']) / position['buy_price']) * 100
                trades.append({
                    'buy_date': position['buy_date'],
                    'sell_date': dates[-1].strftime('%Y-%m-%d'),
                    'buy_price': float(position['buy_price']),
                    'sell_price': float(prices[-1]),
                    'profit_rate': float(profit_rate),
                    'buy_expectation': float(position['buy_expectation']),
                    'sell_expectation': float(expectations[-1])
                })
        
        # 勝率の計算
        if trades:
            winning_trades = len([t for t in trades if t['profit_rate'] > 0])
            total_trades = len(trades)
            win_rate = (winning_trades / total_trades) * 100
            avg_profit = sum(t['profit_rate'] for t in trades) / total_trades
            max_profit = max(t['profit_rate'] for t in trades)
            max_loss = min(t['profit_rate'] for t in trades)
            
            return {
                'win_rate': float(round(win_rate, 2)),
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'avg_profit': float(round(avg_profit, 2)),
                'max_profit': float(round(max_profit, 2)),
                'max_loss': float(round(max_loss, 2)),
                'trades': trades
            }
        
        return None
        
    except Exception as e:
        print(f"バックテスト中にエラーが発生しました: {e}")
        return None

def analyze_single_stock(symbol):
    """
    1銘柄の分析を実行
    """
    # 日本株の場合は.Tを付加
    if not symbol.endswith('.T'):
        symbol = f"{symbol}.T"
        
    hist_data = get_stock_data(symbol)
    if hist_data is None or len(hist_data) < 20:
        return None
        
    try:
        # RSIを計算
        rsi = calculate_rsi(hist_data)
        if rsi is None:
            return None
        
        # ボリンジャーバンドを計算
        upper_band, lower_band, deviation_upper, deviation_lower = calculate_bollinger_bands(hist_data)
        if lower_band is None:
            return None
        
        # 全期間の期待値を計算
        expected_values = []
        for i in range(len(hist_data)):
            if i < 20:  # 20日分のデータが必要
                continue
                
            current_rsi = rsi.iloc[i]
            current_lower_deviation = deviation_lower.iloc[i]
            
            # RSIコンポーネント
            rsi_component = 50 - current_rsi
            
            # 下方乖離率コンポーネント
            deviation_component = abs(min(0, current_lower_deviation))
            
            # 期待値の計算
            raw_expectation = rsi_component + deviation_component
            
            # 期待値の最終調整
            expected_value = round(raw_expectation * 2, 2) if raw_expectation >= 0 else round(raw_expectation, 2)
            expected_values.append(expected_value)
        
        if not expected_values:
            return None
            
        # 最新の期待値、最大値、最小値を取得
        current_expected_value = expected_values[-1]
        max_expected_value = max(expected_values)
        min_expected_value = min(expected_values)
        
        if current_expected_value >= 30:
            stock_name = get_stock_name(symbol)
            
            # バックテストの実行
            backtest_result = calculate_backtest(hist_data, expected_values)
            
            return {
                'symbol': symbol,
                'name': stock_name,
                'expected_value': current_expected_value,
                'max_expected_value': max_expected_value,
                'min_expected_value': min_expected_value,
                'current_price': hist_data['Close'].iloc[-1],
                'ma20': hist_data['Close'].rolling(window=20).mean().iloc[-1],
                'backtest': backtest_result
            }
    except Exception as e:
        return None
    
    return None

def analyze_stocks(symbols):
    """
    銘柄リストを並列処理で分析し、期待値が+30以上の銘柄を抽出
    """
    results = []
    
    # 最大10個のスレッドで並列処理
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # 進捗バーを表示
        futures = list(tqdm(
            executor.map(analyze_single_stock, symbols),
            total=len(symbols),
            desc="銘柄分析中"
        ))
        
        # 結果を収集
        for result in futures:
            if result is not None:
                results.append(result)
    
    # 期待値で降順ソート
    results.sort(key=lambda x: x['expected_value'], reverse=True)
    return results

if __name__ == "__main__":
    # CSVファイルから銘柄リストを読み込む
    df = pd.read_csv('/Users/a0000/上昇期待値指数/prime_list.csv')
    symbols = df['コード'].astype(str).tolist()
    
    print(f"分析対象銘柄数: {len(symbols)}")
    print("分析を開始します...")
    
    start_time = datetime.now()
    results = analyze_stocks(symbols)
    end_time = datetime.now()
    
    print(f"\n分析完了: {len(results)}件の銘柄が見つかりました")
    print(f"分析対象銘柄数: {len(symbols)}件")
    print(f"処理時間: {(end_time - start_time).total_seconds():.1f}秒")
    
    print("\n期待値が+30以上の銘柄:")
    print("=" * 70)
    for result in results:
        print(f"銘柄コード: {result['symbol']}")
        print(f"銘柄名: {result['name']}")
        print(f"期待値: {result['expected_value']:.2f}%")
        print(f"最大期待値: {result['max_expected_value']:.2f}%")
        print(f"最小期待値: {result['min_expected_value']:.2f}%")
        print(f"現在値: {result['current_price']:.2f}")
        print(f"20日移動平均: {result['ma20']:.2f}")
        print("-" * 70) 