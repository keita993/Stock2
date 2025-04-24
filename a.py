import pandas as pd
from datetime import datetime, timedelta

# 対象期間設定（過去1年）
end_date = datetime(2025,4,23)
start_date = end_date - timedelta(days=365)

# SBI証券IPOカレンダーからデータ取得
url = 'https://www.sbisec.co.jp/ETGate/WPLETmgR001Control?getFlg=on&burl=search_domestic&cat1=domestic&cat2=ipo&dir=ipo&file=stock_schedule_ipo.html'
tables = pd.read_html(url)
ipo_df = tables[0]

# 日付フィルタリング
ipo_df['上場日'] = pd.to_datetime(ipo_df['上場日'], format='%Y/%m/%d')
filtered_df = ipo_df[(ipo_df['上場日'] >= start_date) & (ipo_df['上場日'] <= end_date)]
