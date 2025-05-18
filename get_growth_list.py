import pandas as pd

def get_growth_stocks():
    try:
        # Excelファイルを読み込む
        df = pd.read_excel('/Users/a0000/上昇期待値指数/data_j.xls')
        
        # 「グロース（内国株式）」のみを抽出
        growth_stocks = df[df['市場・商品区分'] == 'グロース（内国株式）']
        
        # 必要な列のみを選択
        result_df = growth_stocks[['コード', '銘柄名']].copy()
        
        # コードを文字列型に変換し、ゼロ埋め（4桁）でソート
        result_df['コード'] = result_df['コード'].astype(str).str.zfill(4)
        result_df = result_df.sort_values('コード')
        
        # CSVファイルに保存
        result_df.to_csv('growth_list.csv', index=False, encoding='utf-8')
        print(f"\n処理完了: {len(result_df)}件の銘柄を保存しました")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    get_growth_stocks() 