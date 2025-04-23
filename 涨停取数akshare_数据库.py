import akshare as ak
import pandas as pd
import time  # 用于延时操作
import random  # 用于随机选择User-Agent
import sqlite3
from requests.exceptions import ConnectionError, Timeout, RequestException  # 用于异常处理

# 设置数据库路径
db_path = r"W:\PyFiles\Datas_akshare\market_data.db"

# 获取交易日期历史数据
trade_date_df = ak.tool_trade_date_hist_sina()
trade_date_df['trade_date'] = pd.to_datetime(trade_date_df['trade_date'])

# 只获取今天及之前的交易日期
today = pd.Timestamp.today().normalize()
trade_date_df = trade_date_df[trade_date_df['trade_date'] <= today]

# 检查数据库中已有的数据日期
try:
    conn = sqlite3.connect(db_path)
    # 检查表是否存在
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_data'")
    table_exists = cursor.fetchone() is not None
    
    if table_exists:
        # 获取数据库中最新的交易日期
        latest_date_query = "SELECT MAX(交易日期) FROM market_data"
        latest_date = pd.read_sql(latest_date_query, conn).iloc[0, 0]
        
        if latest_date:
            latest_date = pd.to_datetime(latest_date)
            print(f"数据库中最新数据日期: {latest_date.strftime('%Y-%m-%d')}")
            
            # 只获取比数据库中最新日期更新的交易日期
            # 注意：这里只获取新的日期数据，不会删除旧数据
            recent_dates = trade_date_df[trade_date_df['trade_date'] > latest_date]
            
            if recent_dates.empty:
                print("数据库已是最新，无需更新数据")
                conn.close()
                exit(0)
            else:
                print(f"将获取从 {recent_dates['trade_date'].min().strftime('%Y-%m-%d')} 到 {recent_dates['trade_date'].max().strftime('%Y-%m-%d')} 的新数据")
                # 如果新数据超过30天，只取最近30天
                if len(recent_dates) > 30:
                    recent_30_dates = recent_dates.tail(30)
                    print(f"新数据超过30天，只获取最近30天数据")
                else:
                    recent_30_dates = recent_dates
        else:
            # 数据库表存在但没有数据
            print("数据库表存在但没有数据，将获取最近30个交易日的数据")
            recent_30_dates = trade_date_df.tail(30)
    else:
        # 数据库表不存在
        print("数据库表不存在，将获取最近30个交易日的数据")
        recent_30_dates = trade_date_df.tail(30)
    
    conn.close()
except Exception as e:
    print(f"检查数据库时出错: {str(e)}")
    print("将获取最近30个交易日的数据")
    recent_30_dates = trade_date_df.tail(30)

# 用于存储所有类型的数据 - 使用列表而非DataFrame以减少内存占用
zt_data = []  # 涨停数据
dt_data = []  # 跌停数据
zb_data = []  # 炸板数据

# 设置请求头，模拟浏览器行为减少被拒绝的可能性
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
]

# 遍历交易日获取数据
for date in recent_30_dates['trade_date']:
    date_str = date.strftime('%Y%m%d')
    print(f"正在获取 {date_str} 的数据...")
    
    # 改进的重试机制
    max_retries = 10
    retry_delay = 2
    max_delay = 60
    
    # 定义获取数据的函数，以便于重试
    def fetch_data(date_str, data_type):
        if data_type == 'zt':
            return ak.stock_zt_pool_em(date=date_str)
        elif data_type == 'dt':
            return ak.stock_zt_pool_dtgc_em(date=date_str)
        elif data_type == 'zb':
            return ak.stock_zt_pool_zbgc_em(date=date_str)
        return pd.DataFrame()
    
    # 为每种数据类型分别获取数据
    data_types = [('zt', zt_data, '涨停'), ('dt', dt_data, '跌停'), ('zb', zb_data, '炸板')]
    
    for data_code, data_list, data_name in data_types:
        for attempt in range(max_retries):
            try:
                # 随机选择一个User-Agent
                ak.requests_headers = {'User-Agent': random.choice(user_agents)}
                
                # 获取数据
                daily_data = fetch_data(date_str, data_code)
                
                if not daily_data.empty:
                    # 只添加必要的列，减少内存占用
                    daily_data['交易日期'] = date
                    daily_data['数据类型'] = data_name
                    data_list.append(daily_data)
                
                # 成功获取数据，跳出重试循环
                break
                
            except (ConnectionError, Timeout, RequestException) as e:
                if attempt < max_retries - 1:
                    # 计算退避时间，但不超过最大延迟
                    current_delay = min(retry_delay * (2 ** attempt), max_delay)
                    # 添加随机抖动，避免同步请求
                    wait_time = current_delay + (current_delay * 0.2)
                    
                    print(f"获取{data_name}数据时连接错误: {str(e)}，{wait_time:.2f}秒后重试... (尝试 {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"获取{date_str}的{data_name}数据失败，已达到最大重试次数: {str(e)}")
            except Exception as e:
                print(f"获取{date_str}的{data_name}数据时发生错误: {str(e)}")
                break  # 对于非连接错误，不进行重试
        
        # 每种数据类型之间添加短暂延时
        time.sleep(0.5)
    
    # 每个日期之间添加延时
    time.sleep(1)

# 检查是否获取到任何数据
if not any([zt_data, dt_data, zb_data]):
    print("Error: 未能获取任何数据")
else:
    # 合并所有类型的数据
    all_dfs = []
    if zt_data:
        all_dfs.append(pd.concat(zt_data, ignore_index=True))
    if dt_data:
        all_dfs.append(pd.concat(dt_data, ignore_index=True))
    if zb_data:
        all_dfs.append(pd.concat(zb_data, ignore_index=True))
    
    # 合并所有数据
    filtered_df = pd.concat(all_dfs, ignore_index=True)
    
    # 按日期和数据类型排序
    filtered_df = filtered_df.sort_values(['交易日期', '数据类型'], ascending=[False, True])
    
    # 打印数据基本信息
    print("数据统计概览:")
    print("数据形状:", filtered_df.shape)
    print("\n数据类型分布:")
    print(filtered_df['数据类型'].value_counts())
    print("\n数据列名:", filtered_df.columns.tolist())
    print("\n统计日期范围:")
    print("开始日期:", filtered_df['交易日期'].min().strftime('%Y-%m-%d'))
    print("结束日期:", filtered_df['交易日期'].max().strftime('%Y-%m-%d'))
    
    # 尝试保存数据到Excel
    try:
        output_file = r"W:\PyFiles\Datas_akshare\market_data_analysis.xlsx"
        filtered_df.to_excel(output_file, index=False)
        print("\n数据已成功保存到", output_file)
    except PermissionError:
        print("\n错误：无法保存文件，可能是文件被占用或没有写入权限。请关闭已打开的Excel文件后重试。")
    except Exception as e:
        print("\n保存Excel文件时发生错误:", str(e))
    
    # 保存数据到SQLite数据库
    try:
        # 连接到SQLite数据库（如果不存在则创建）
        conn = sqlite3.connect(db_path)
        
        # 将日期列转换为字符串格式，避免SQLite存储问题
        filtered_df['交易日期'] = filtered_df['交易日期'].dt.strftime('%Y-%m-%d')
        
        # 检查表是否存在
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_data'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            # 表存在，追加新数据
            # 先删除可能重复的日期数据，以避免重复
            # 注意：这里只删除新获取数据对应日期的记录，不会删除历史数据
            for date in filtered_df['交易日期'].unique():
                conn.execute(f"DELETE FROM market_data WHERE 交易日期 = '{date}'")
            
            # 追加新数据到现有表，保留历史数据
            filtered_df.to_sql('market_data', conn, if_exists='append', index=False)
            print(f"\n新数据已追加到SQLite数据库: {db_path}")
            
            # 获取数据库中的总记录数
            total_records = pd.read_sql("SELECT COUNT(*) FROM market_data", conn).iloc[0, 0]
            print(f"数据库中现有记录总数: {total_records}")
            
            # 获取数据库中的日期范围
            date_range = pd.read_sql("SELECT MIN(交易日期), MAX(交易日期) FROM market_data", conn)
            print(f"数据库中数据日期范围: {date_range.iloc[0, 0]} 至 {date_range.iloc[0, 1]}")
        else:
            # 表不存在，创建新表
            filtered_df.to_sql('market_data', conn, if_exists='replace', index=False)
            print(f"\n数据已成功保存到新建的SQLite数据库: {db_path}")
        
        # 创建索引以提高查询性能
        conn.execute('CREATE INDEX IF NOT EXISTS idx_date ON market_data(交易日期)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_type ON market_data(数据类型)')
        
        # 关闭连接
        conn.close()
        
        print("您可以使用SQLite工具或Python代码查询此数据库")
    except Exception as e:
        print("\n保存到数据库时发生错误:", str(e))