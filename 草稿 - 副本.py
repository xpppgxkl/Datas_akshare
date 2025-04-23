import akshare as ak
import datetime
import time
import pandas as pd

# 设置显示所有列
pd.set_option('display.max_columns', None)

# 设置显示所有行
pd.set_option('display.max_rows', None)

# 设置宽度，避免列被截断
pd.set_option('display.width', None)

# 设置单元格内容的显示宽度
pd.set_option('display.max_colwidth', None)

db_path = r"W:\PyFiles\Datas_akshare\market_data_analysis_20250420_193426.csv"
df = pd.read_csv(db_path)
print(df)
