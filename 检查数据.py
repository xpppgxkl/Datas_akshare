import akshare as ak
import pandas as pd

# 获取乐咕市场活动数据
df = ak.stock_market_activity_legu()

# 打印数据基本信息
print('数据形状:', df.shape)
print('\n数据类型:')
print(df.dtypes)
print('\n数据前10行:')
print(df.head(10))
print('\n数据唯一值:')
print(df['item'].unique())

# 获取交易日期数据表
tool_trade_date_hist_sina_df = ak.tool_trade_date_hist_sina()

# 获取最近的30个交易日期
tool_trade_date_hist_sina_df['trade_date'] = pd.to_datetime(tool_trade_date_hist_sina_df['trade_date'])
recent_30_trade_dates = tool_trade_date_hist_sina_df[tool_trade_date_hist_sina_df['trade_date'] <= pd.Timestamp.today()].tail(30)

# 打印交易日期信息
print('\n最近30个交易日期:')
print(recent_30_trade_dates['trade_date'].dt.strftime('%Y-%m-%d').tolist())

# 尝试转换item列为日期格式并打印结果
print('\n尝试转换item列为日期格式:')
df['item_date'] = pd.to_datetime(df['item'], errors='coerce')
print(df[['item', 'item_date']].head(10))