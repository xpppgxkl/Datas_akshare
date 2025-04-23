import pandas as pd
import numpy as np
from openpyxl.styles import Alignment
import gc  # 垃圾回收模块
import warnings
import os  # 文件操作模块
from functools import lru_cache  # 缓存装饰器
from time import time  # 性能测量
import concurrent.futures  # 多线程/多进程处理

# 尝试导入可选模块
try:
    import psutil  # 内存监控
except ImportError:
    print("提示: psutil模块未安装，内存监控功能将被禁用")
    psutil = None

try:
    import swifter  # 并行处理优化
    has_swifter = True
except ImportError:
    print("提示: swifter模块未安装，相关优化将被禁用")
    has_swifter = False

# 忽略pandas的警告
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# 配置参数
CONFIG = {
    'chunk_size': 50000,  # 每块数据的行数
    'max_workers': max(4, os.cpu_count() - 1) if os.cpu_count() else 4,  # 工作线程数量，留一个核心给系统
    'memory_threshold': 0.85,  # 内存使用阈值，超过则强制垃圾回收
    'use_swifter': has_swifter,  # 是否使用swifter加速
    'use_process_pool': False,  # 是否使用进程池而非线程池 (True 适合计算密集型, False 适合IO密集型)
}

# 内存监控和垃圾回收函数
def check_memory_usage():
    """监控内存使用情况，如果超过阈值则进行垃圾回收"""
    if psutil:
        mem_usage = psutil.virtual_memory().percent / 100
        if mem_usage > CONFIG['memory_threshold']:
            gc.collect()
            return True
    else:
        # 如果没有psutil，定期进行垃圾回收
        gc.collect()
    return False

# 装饰器：执行时间统计
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f"{func.__name__} 执行完成，耗时: {end_time - start_time:.2f}秒")
        return result
    return wrapper

# 多线程处理数据的函数
def process_data_chunk(chunk, col_name=None):
    """处理数据块的函数，将在线程池中执行"""
    # 确保日期列为datetime类型
    if 'processed' not in chunk.columns:
        chunk = chunk.copy()
        chunk['交易日期'] = pd.to_datetime(chunk['交易日期'])
        chunk['processed'] = True
    return chunk

# 多线程计算热点板块
def calculate_hot_industries(date, industry_date_counts):
    """计算单个日期的热点板块 - 线程安全的函数"""
    if date not in industry_date_counts.index:
        return {date: {'热点板块1': '无', '热点板块2': '无'}}
    
    # 获取当前日期的行业计数
    date_counts = industry_date_counts.loc[date]
    
    # 如果没有数据，设置为无
    if date_counts.sum() == 0:
        return {date: {'热点板块1': '无', '热点板块2': '无'}}
    
    # 获取前两名行业
    top_two = date_counts.nlargest(2)
    result = [f"{industry}({count})" for industry, count in top_two.items()]
    
    # 确保始终返回两个值
    if len(result) == 0:
        return {date: {'热点板块1': '无', '热点板块2': '无'}}
    elif len(result) == 1:
        return {date: {'热点板块1': result[0], '热点板块2': '无'}}
    else:
        return {date: {'热点板块1': result[0], '热点板块2': result[1]}}

# 多线程计算涨停梯队
def calculate_zt_tiers(date, zt_df, col_name):
    """计算单个日期的涨停梯队 - 线程安全的函数"""
    date_data = zt_df[zt_df['交易日期'] == date]
    if len(date_data) == 0:
        return {date: '无连板'}
        
    # 获取最高连板数
    max_tier = int(date_data[col_name].max()) if not date_data.empty else 0
    if max_tier < 1:
        return {date: '无连板'}
        
    # 修改：统计所有连板数据，从最高板一直到1板
    min_tier = 1  # 修改为固定从1板开始统计
    
    # 使用value_counts更高效地统计
    tier_counts = date_data[col_name].value_counts()
    
    # 格式化输出，跳过数量为0的板块
    result = []
    for tier in range(max_tier, min_tier - 1, -1):
        count = tier_counts.get(tier, 0)
        if count > 0:
            result.append(f"{tier}{{{count}}}")
    
    return {date: "".join(result) if result else '无连板'}

# 多线程计算高度板
def calculate_high_tier_stocks(date, zt_df, col_name):
    """计算单个日期的高度板 - 线程安全的函数"""
    if col_name is None or '代码' not in zt_df.columns:
        return {date: '无'}
        
    date_data = zt_df[zt_df['交易日期'] == date]
    if len(date_data) == 0:
        return {date: '无'}
        
    # 获取最高连板数
    max_tier = date_data[col_name].max() if not date_data.empty else 0
    if max_tier < 1:
        return {date: '无'}
        
    # 使用字典存储每个tier的股票，避免重复过滤数据
    result_stocks = []
    current_tier = max_tier
    
    # 最多处理3个tier级别
    while current_tier > 1 and len(result_stocks) < 3:
        # 获取当前连板数的所有股票
        current_stocks = date_data[date_data[col_name] == current_tier]
        if not current_stocks.empty:
            # 使用列表推导式更高效地格式化股票代码
            codes = [f"{str(code).zfill(6)}[{int(current_tier)}]" for code in current_stocks['代码']]
            result_stocks.append(';'.join(codes))
        
        current_tier -= 1
    
    return {date: ' '.join(result_stocks) if result_stocks else '无'}

# 多线程计算连板晋级率
@lru_cache(maxsize=1024)  # 增大缓存容量
def calculate_tier_promotion_rates(today_date, yesterday_date, zt_df_hash, col_name):
    """计算连板级别晋级率 - 添加zt_df的哈希值作为缓存键的一部分"""
    if col_name is None:
        return ""
    
    # 获取今天和昨天的数据
    today_data = zt_df[zt_df['交易日期'] == today_date]
    yesterday_data = zt_df[zt_df['交易日期'] == yesterday_date]
    
    if today_data.empty or yesterday_data.empty:
        return ""
    
    # 获取今天的最高连板数
    max_tier = int(today_data[col_name].max())
    if max_tier < 2:  # 至少需要2板才有晋级率
        return ""
    
    # 使用value_counts一次性计算所有级别的数量
    today_tiers = today_data[col_name].value_counts().to_dict()
    yesterday_tiers = yesterday_data[col_name].value_counts().to_dict()
    
    # 计算所有2板及以上的连板级别的晋级率
    result_parts = []
    
    # 从最高板开始，一直计算到2板
    for tier in range(max_tier, 1, -1):
        # 今天tier板的数量
        today_count = today_tiers.get(tier, 0)
        if today_count == 0:
            continue
            
        # 昨天(tier-1)板的数量
        yesterday_count = yesterday_tiers.get(tier-1, 0)
        
        # 计算晋级率
        if yesterday_count > 0:
            rate = (today_count / yesterday_count) * 100
            result_parts.append(f"{tier}[{rate:.0f}%]")
    
    # 计算总体晋级率
    today_consecutive_count = sum(count for tier, count in today_tiers.items() if tier >= 2)
    yesterday_total_count = sum(yesterday_tiers.values())
    
    if yesterday_total_count > 0:
        total_rate = (today_consecutive_count / yesterday_total_count) * 100
        result_parts.append(f"总[{total_rate:.0f}%]")
    
    return "".join(result_parts)

try:
    # 记录开始时间
    total_start_time = time()
    print(f"系统配置: CPU核心数={os.cpu_count()}, 使用线程数={CONFIG['max_workers']}")
    
    # 读取CSV文件 - 只读取必要的列以减少内存使用
    usecols = ['交易日期', '数据类型', '所属行业', '代码']
    csv_file = r"W:\PyFiles\Datas_akshare\market_data_analysis.csv"
    
    print("开始读取数据文件...")
    read_start_time = time()
    
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"数据文件不存在: {csv_file}")
        
    # 获取文件大小
    file_size = os.path.getsize(csv_file) / (1024 * 1024)  # 文件大小（MB）
    print(f"数据文件大小: {file_size:.2f}MB")
    
    # 检查连板数列 - 先读取列名
    temp_df = pd.read_csv(csv_file, nrows=0)
    if '连续涨停天数' in temp_df.columns:
        usecols.append('连续涨停天数')
        col_name = '连续涨停天数'
    elif '连板数' in temp_df.columns:
        usecols.append('连板数')
        col_name = '连板数'
    else:
        col_name = None
    
    # 优化：设置数据类型以减少内存使用
    dtype_dict = {
        '代码': 'str',
        '所属行业': 'category',  # 使用category类型减少内存
        '数据类型': 'category'
    }
    
    # 读取完整数据，使用分块读取策略
    if file_size > 50:  # 如果文件较大，使用分块读取和多线程处理
        print(f"文件较大 ({file_size:.1f}MB)，使用分块读取和多线程处理...")
        
        # 创建线程池或进程池
        executor_class = concurrent.futures.ProcessPoolExecutor if CONFIG['use_process_pool'] else concurrent.futures.ThreadPoolExecutor
        
        chunks = []
        processed_rows = 0
        
        # 使用with语句确保资源释放
        with executor_class(max_workers=CONFIG['max_workers']) as executor:
            futures = []
            
            # 分块读取和处理
            for i, chunk in enumerate(pd.read_csv(csv_file, usecols=usecols, dtype=dtype_dict, chunksize=CONFIG['chunk_size'])):
                # 提交处理任务到线程池
                futures.append(executor.submit(process_data_chunk, chunk, col_name))
                processed_rows += len(chunk)
                
                # 显示进度
                print(f"已读取并提交处理 {processed_rows} 行数据...", end='\r')
                
                # 如果积累了足够多的future对象，开始收集结果
                if len(futures) >= CONFIG['max_workers'] * 2:
                    for future in concurrent.futures.as_completed(futures):
                        chunks.append(future.result())
                    futures = []
                    
                    # 内存监控
                    check_memory_usage()
            
            # 收集剩余的结果
            print("\n等待所有数据块处理完成...")
            for future in concurrent.futures.as_completed(futures):
                chunks.append(future.result())
        
        print(f"\n所有数据块处理完成，正在合并数据...")
        df = pd.concat(chunks, ignore_index=True)
        
        # 释放内存
        del chunks
        gc.collect()
    else:
        # 对于小文件，直接读取
        df = pd.read_csv(csv_file, usecols=usecols, dtype=dtype_dict)
    
    # 确保日期列为datetime类型 - 如果之前的处理没有转换
    if not pd.api.types.is_datetime64_any_dtype(df['交易日期']):
        df['交易日期'] = pd.to_datetime(df['交易日期'])
    
    print(f"数据读取完成，耗时: {time() - read_start_time:.2f}秒")
    print(f"数据行数: {len(df)}, 列数: {len(df.columns)}")
    print(f"内存占用: {df.memory_usage(deep=True).sum() / (1024*1024):.2f}MB")
    
    # 主动触发垃圾回收
    gc.collect()
    
    # 检查必要的列是否存在
    required_columns = ['交易日期', '数据类型', '所属行业']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f'缺少必要的列：{", ".join(missing_cols)}')
    
    # 记录数据过滤开始时间
    filter_start_time = time()
    
    # 使用字典存储不同类型的数据
    data_types = {'涨停': None, '跌停': None, '炸板': None}
    
    # 优化：一次性过滤数据
    for data_type in data_types.keys():
        mask = df['数据类型'] == data_type
        data_types[data_type] = df[mask].copy()
    
    # 分配过滤后的数据
    zt_df = data_types['涨停']
    dt_df = data_types['跌停']
    zb_df = data_types['炸板']
    
    # 释放不再需要的数据
    del data_types
    del df
    
    print(f"数据过滤完成，耗时: {time() - filter_start_time:.2f}秒")
    print(f"涨停数据: {len(zt_df)}行, 跌停数据: {len(dt_df)}行, 炸板数据: {len(zb_df)}行")
    
    # 主动触发垃圾回收
    gc.collect()
    
    # 获取所有日期并创建日期索引 - 使用集合操作更高效
    all_dates = set()
    if not zt_df.empty:
        all_dates.update(zt_df['交易日期'])
    if not dt_df.empty:
        all_dates.update(dt_df['交易日期'])
    if not zb_df.empty:
        all_dates.update(zb_df['交易日期'])
    
    # 创建日期索引
    daily_stats = pd.DataFrame(index=pd.DatetimeIndex(sorted(all_dates)))
    
    # 统计开始
    stats_start_time = time()
    
    # 使用预先过滤的数据进行统计 - 一次性计算所有计数
    counts_dict = {}
    
    # 并行计算基础统计量
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        future_to_key = {}
        
        # 提交涨停数统计任务
        if not zt_df.empty:
            future_to_key[executor.submit(lambda df: df['交易日期'].value_counts(), zt_df)] = '涨停数'
        else:
            counts_dict['涨停数'] = pd.Series(0, index=daily_stats.index)
            
        # 提交跌停数统计任务
        if not dt_df.empty:
            future_to_key[executor.submit(lambda df: df['交易日期'].value_counts(), dt_df)] = '跌停数'
        else:
            counts_dict['跌停数'] = pd.Series(0, index=daily_stats.index)
            
        # 提交炸板数统计任务
        if not zb_df.empty:
            future_to_key[executor.submit(lambda df: df['交易日期'].value_counts(), zb_df)] = '炸板数'
        else:
            counts_dict['炸板数'] = pd.Series(0, index=daily_stats.index)
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                counts_dict[key] = future.result()
            except Exception as e:
                print(f"计算{key}时发生错误: {str(e)}")
                counts_dict[key] = pd.Series(0, index=daily_stats.index)
    
    # 将计数添加到daily_stats
    daily_stats = daily_stats.assign(**counts_dict)
    
    print(f"基础统计计算完成，耗时: {time() - stats_start_time:.2f}秒")
    
    # 主动触发垃圾回收
    gc.collect()
    
    # 热点板块统计 - 使用多线程优化
    start_time = time()
    
    if not zt_df.empty:
        # 使用crosstab一次性计算所有日期和行业的组合计数
        industry_date_counts = pd.crosstab(zt_df['交易日期'], zt_df['所属行业'])
        
        # 使用多线程并行计算所有日期的热点板块
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
            # 提交所有日期的计算任务
            future_to_date = {
                executor.submit(calculate_hot_industries, date, industry_date_counts): date 
                for date in daily_stats.index
            }
            
            # 创建结果字典
            hot_industries = {}
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_date):
                try:
                    hot_industries.update(future.result())
                except Exception as e:
                    date = future_to_date[future]
                    print(f"计算{date}热点板块时发生错误: {str(e)}")
                    hot_industries[date] = {'热点板块1': '无', '热点板块2': '无'}
    else:
        # 如果没有涨停数据，所有日期的热点板块都是'无'
        hot_industries = {date: {'热点板块1': '无', '热点板块2': '无'} for date in daily_stats.index}
    
    # 将热点板块数据添加到daily_stats
    hot_industries_df = pd.DataFrame.from_dict(hot_industries, orient='index')
    daily_stats = daily_stats.join(hot_industries_df)
    
    print(f"热点板块统计耗时: {time() - start_time:.2f}秒")
    
    # 计算连板股票的最高版数 - 优化计算方式
    if col_name and not zt_df.empty:
        # 优化：直接使用groupby计算最大值
        max_tier_by_date = zt_df.groupby('交易日期')[col_name].max()
        daily_stats['最高连板数'] = max_tier_by_date.reindex(daily_stats.index).fillna(0).astype(int)
    else:
        daily_stats['最高连板数'] = 0
    
    # 统计涨停梯队 - 使用多线程优化
    if col_name and not zt_df.empty:
        start_time = time()
        
        # 使用多线程并行计算所有日期的涨停梯队
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
            # 提交所有日期的计算任务
            future_to_date = {
                executor.submit(calculate_zt_tiers, date, zt_df, col_name): date 
                for date in daily_stats.index
            }
            
            # 创建结果字典
            zt_tiers = {}
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_date):
                try:
                    zt_tiers.update(future.result())
                except Exception as e:
                    date = future_to_date[future]
                    print(f"计算{date}涨停梯队时发生错误: {str(e)}")
                    zt_tiers[date] = '无连板'
    else:
        # 如果没有连板数列，所有日期的涨停梯队都是'无连板'
        zt_tiers = {date: '无连板' for date in daily_stats.index}
    
    # 将涨停梯队添加到daily_stats
    daily_stats['涨停梯队'] = pd.Series(zt_tiers)
    
    print(f"涨停梯队统计耗时: {time() - start_time:.2f}秒")
    
    # 计算高度板 - 使用多线程优化
    if col_name and '代码' in zt_df.columns and not zt_df.empty:
        start_time = time()
        
        # 使用多线程并行计算所有日期的高度板
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
            # 提交所有日期的计算任务
            future_to_date = {
                executor.submit(calculate_high_tier_stocks, date, zt_df, col_name): date 
                for date in daily_stats.index
            }
            
            # 创建结果字典
            high_tier_stocks = {}
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_date):
                try:
                    high_tier_stocks.update(future.result())
                except Exception as e:
                    date = future_to_date[future]
                    print(f"计算{date}高度板时发生错误: {str(e)}")
                    high_tier_stocks[date] = '无'
    else:
        # 如果没有连板数列或代码列，所有日期的高度板都是'无'
        high_tier_stocks = {date: '无' for date in daily_stats.index}
    
    # 将高度板添加到daily_stats
    daily_stats['高度板'] = pd.Series(high_tier_stocks)
    
    # 计算今日连板数（连板数>=2的股票数量）- 优化版本
    if col_name and not zt_df.empty:
        # 使用groupby和聚合函数一次性计算
        consecutive_counts = zt_df[zt_df[col_name] >= 2].groupby('交易日期').size()
        daily_stats['今日连板数'] = consecutive_counts.reindex(daily_stats.index, fill_value=0)
    else:
        daily_stats['今日连板数'] = 0
    
    # 计算昨日涨停数和晋级率 - 优化版本
    if not zt_df.empty:
        # 获取所有日期并排序
        all_dates = sorted(zt_df['交易日期'].unique())
        
        # 使用Series.value_counts一次性计算所有日期的涨停数量
        date_to_zt_count = zt_df['交易日期'].value_counts()
        
        # 创建昨日涨停数和晋级率的Series
        yesterday_zt_counts = pd.Series(index=all_dates, dtype=float)
        tier_promotion_rates = pd.Series(index=all_dates, dtype=str)
        
        # 创建zt_df的哈希值，用于缓存键
        zt_df_hash = hash(str(len(zt_df)))
        
        # 并行计算晋级率
        if col_name:
            with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
                future_to_idx = {}
                
                # 跳过第一个日期，因为它没有前一天的数据
                for i in range(1, len(all_dates)):
                    today = all_dates[i]
                    yesterday = all_dates[i-1]
                    
                    # 提交计算任务
                    future = executor.submit(
                        calculate_tier_promotion_rates, 
                        today, yesterday, zt_df_hash, col_name
                    )
                    future_to_idx[future] = i
                
                # 收集结果
                for future in concurrent.futures.as_completed(future_to_idx):
                    i = future_to_idx[future]
                    today = all_dates[i]
                    yesterday = all_dates[i-1]
                    
                    try:
                        # 获取晋级率结果
                        tier_promotion_rates[today] = future.result()
                        
                        # 昨日涨停总数
                        yesterday_count = date_to_zt_count.get(yesterday, 0)
                        yesterday_zt_counts[today] = yesterday_count
                    except Exception as e:
                        print(f"计算{today}晋级率时发生错误: {str(e)}")
                        tier_promotion_rates[today] = ""
                        yesterday_zt_counts[today] = 0
    
    # 将昨日涨停数和晋级率添加到daily_stats
    daily_stats['_昨日涨停数'] = yesterday_zt_counts  # 添加下划线前缀，用于内部计算
    daily_stats['晋级率'] = tier_promotion_rates  # 使用新的各级别晋级率
    
    # 记录最终处理开始时间
    final_process_start_time = time()
    
    # 统计行业分布 - 只计算一次
    if not zt_df.empty:
        industry_counts = zt_df['所属行业'].value_counts().head(10)
    else:
        industry_counts = pd.Series(dtype='int64')
    
    # 填充缺失值为0 - 优化版本
    # 只填充数值列
    numeric_cols = daily_stats.select_dtypes(include=['number']).columns
    daily_stats[numeric_cols] = daily_stats[numeric_cols].fillna(0)
    
    # 按日期降序排序
    daily_stats = daily_stats.sort_index(ascending=False)
    
    # 将索引（日期）转换为字符串格式，只保留年月日
    daily_stats.index = daily_stats.index.strftime('%Y-%m-%d')
    
    # 在输出到Excel之前删除内部计算列
    if '_昨日涨停数' in daily_stats.columns:
        output_stats = daily_stats.drop(columns=['_昨日涨停数'])
    else:
        output_stats = daily_stats
    
    # 重新排序列，将涨停梯队和高度板放在最后
    special_cols = ['涨停梯队', '高度板']
    cols = [col for col in output_stats.columns if col not in special_cols]
    # 只添加实际存在的特殊列
    for col in special_cols:
        if col in output_stats.columns:
            cols.append(col)
    
    # 只保留需要的列
    output_stats = output_stats[cols]
    
    print(f"最终数据处理完成，耗时: {time() - final_process_start_time:.2f}秒")
    
    # 打印结果 - 使用更简洁的方式
    print(f'\n每日统计数据（{len(output_stats)}行）：')
    if not output_stats.empty:
        print(output_stats.head())
        print('...')
    else:
        print("无数据")
    
    print('\n出现最多的行业（前10）：')
    print(industry_counts if not industry_counts.empty else "无数据")
    
    # 输出内存使用情况
    print("\n内存使用情况:")
    for var_name, var in {'output_stats': output_stats, 'zt_df': zt_df, 'dt_df': dt_df, 'zb_df': zb_df}.items():
        if var is not None:
            print(f"{var_name}: {var.memory_usage(deep=True).sum() / (1024*1024):.2f}MB")
    
    # 保存结果到Excel - 优化文件写入
    start_time = time()
    output_file = '涨停数据统计结果.xlsx'
    
    # 使用优化的Excel写入方式
    try:
        # 检查文件是否已存在且被打开
        if os.path.exists(output_file):
            try:
                # 尝试以独占方式打开文件，如果文件被占用会抛出异常
                with open(output_file, 'r+b') as f:
                    pass
            except IOError:
                print(f"警告: 文件 {output_file} 可能被其他程序占用，将尝试使用临时文件名保存")
                output_file = f'涨停数据统计结果_{int(time())}.xlsx'
        
        # 使用to_excel的优化参数
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
            # 使用float_format参数减少浮点数精度，节省空间
            output_stats.to_excel(writer, sheet_name='Sheet1', float_format='%.1f', index_label='交易日期')
            
            # 获取工作簿和工作表
            worksheet = writer.sheets['Sheet1']
            
            # 设置右对齐的列
            right_align_cols = ['涨停梯队', '晋级率']
            if '高度板' in output_stats.columns:
                right_align_cols.append('高度板')
            
            # 设置右对齐 - 批量设置单元格样式
            for col_name in right_align_cols:
                if col_name in output_stats.columns:
                    col_idx = output_stats.columns.get_loc(col_name) + 2  # +2是因为Excel列从1开始，且第一列是索引
                    # 使用列表推导式一次性创建所有单元格对象
                    cells = [worksheet.cell(row=row, column=col_idx) for row in range(2, len(output_stats) + 2)]
                    # 一次性设置所有单元格的对齐方式
                    for cell in cells:
                        cell.alignment = Alignment(horizontal='right')
        
        print(f"Excel文件写入耗时: {time() - start_time:.2f}秒")
    except Exception as e:
        print(f"保存Excel文件时出错: {str(e)}")
    
    print('\n分析结果已保存到：' + output_file)
    
    # 输出总耗时
    print(f"\n总耗时: {time() - total_start_time:.2f}秒")
    
    # 最终垃圾回收
    gc.collect()

except Exception as e:
    print(f'发生错误：{str(e)}')
    import traceback
    print(traceback.format_exc())  # 打印详细的错误堆栈信息
finally:
    # 确保所有资源被释放
    gc.collect()
    print("程序结束，已清理资源")
    