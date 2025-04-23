import sqlite3
import pandas as pd

# 连接到数据库
db_path = r"W:\PyFiles\Datas_akshare\market_data.db"
conn = sqlite3.connect(db_path)

# 示例1: 查询最近一天的涨停数据
query1 = """
SELECT * FROM market_data 
WHERE 数据类型 = '涨停' 
AND 交易日期 = (SELECT MAX(交易日期) FROM market_data)
"""
df1 = pd.read_sql(query1, conn)
print("最近一天的涨停数据数量:", len(df1))

# 示例2: 查询特定日期范围内的所有类型数据
query2 = """
SELECT 交易日期, 数据类型, COUNT(*) as 数量
FROM market_data
WHERE 交易日期 BETWEEN '2023-01-01' AND '2023-12-31'
GROUP BY 交易日期, 数据类型
ORDER BY 交易日期 DESC
"""
df2 = pd.read_sql(query2, conn)
print("\n2023年数据统计:")
print(df2)

# 示例3: 查询特定股票的所有记录
stock_code = '000001'  # 示例股票代码，请根据实际数据调整
query3 = f"""
SELECT * FROM market_data
WHERE 代码 = '{stock_code}'
ORDER BY 交易日期 DESC
"""
df3 = pd.read_sql(query3, conn)
print(f"\n股票 {stock_code} 的所有记录:")
print(df3)

# 关闭连接
conn.close()