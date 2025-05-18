import pandas as pd
df = pd.read_excel('/Users/a0000/上昇期待値指数/data_j.xls')
print(df['市場・商品区分'].unique())