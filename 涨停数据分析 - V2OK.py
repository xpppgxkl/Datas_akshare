import pandas as pd
import numpy as np
from openpyxl.styles import Alignment
import gc  # 添加垃圾回收模块
import warnings
import os  # 添加os模块用于文件操作
from functools import lru_cache  # 添加缓存装饰器
from time import time  # 添加时间模块用于性能测量

# 忽略pandas的SettingWithCopyWarning警告
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

try:
    # 记录开始时间
    total_start_time = time()
    
    # 读取Excel文件 - 只读取必要的列以减少内存使用
    usecols = ['交易日期', '数据类型', '所属行业', '代码']
    excel_file = r"W:\PyFiles\Datas_akshare\market_data_analysis.xlsx"
    
    print("开始读取数据文件...")
    read_start_time = time()
    
    # 检查文件是否存在
    if not os.path.exists(excel_file):
        raise FileNotFoundError(f"数据文件不存在: {excel_file}")
        
    # 获取文件大小
    file_size = os.path.getsize(excel_file) / (1024 * 1024)  # 文件大小（MB）
    print(f"数据文件大小: {file_size:.2f}MB")
    
    # 检查连板数列 - 只读取列名，不读取数据
    temp_df = pd.read_excel(excel_file, nrows=0)
    if '连续涨停天数' in temp_df.columns:
        usecols.append('连续涨停天数')
        col_name = '连续涨停天数'
    elif '连板数' in temp_df.columns:
        usecols.append('连板数')
        col_name = '连板数'
    else:
        col_name = None
    
    # 读取完整数据，但只读取需要的列，使用chunksize分块读取以减少内存占用
    # 对于较小的数据集，可以直接读取；对于大数据集，考虑分块处理
    file_size = os.path.getsize(excel_file) / (1024 * 1024)  # 文件大小（MB）
    
    if file_size > 100:  # 如果文件大于100MB，使用分块读取
        print(f"文件较大 ({file_size:.1f}MB)，使用分块读取...")
        chunks = []
        chunk_size = 10000
        # 使用engine='openpyxl'可能对某些Excel文件更高效
        for chunk in pd.read_excel(excel_file, usecols=usecols, chunksize=chunk_size):
            chunks.append(chunk)
            # 显示进度
            print(f"已读取 {len(chunks) * chunk_size} 行数据...", end='\r')
        df = pd.concat(chunks, ignore_index=True)
        print("\n分块读取完成，正在合并数据...")
        # 释放内存
        del chunks
        gc.collect()
    else:
        # 对于小文件，直接读取
        df = pd.read_excel(excel_file, usecols=usecols)
    
    print(f"数据读取完成，耗时: {time() - read_start_time:.2f}秒")
    print(f"数据行数: {len(df)}, 列数: {len(df.columns)}")
    
    # 主动触发垃圾回收
    gc.collect()
    
    # 检查必要的列是否存在
    required_columns = ['交易日期', '数据类型', '所属行业']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f'缺少必要的列：{", ".join(missing_cols)}')
    
    # 确保日期列为datetime类型
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    
    # 记录数据过滤开始时间
    filter_start_time = time()
    
    # 使用字典存储不同类型的数据，避免多次过滤
    data_types = {'涨停': None, '跌停': None, '炸板': None}
    
    # 使用更高效的方式过滤数据
    # 对于大数据集，使用query可能更高效；对于小数据集，布尔索引可能更快
    if len(df) > 100000:  # 大数据集
        print("使用query方法过滤大数据集...")
        for data_type in data_types.keys():
            data_types[data_type] = df.query(f"数据类型 == '{data_type}'")
    else:  # 小数据集
        # 一次性创建数据类型的映射，避免重复计算
        data_type_series = df['数据类型']
        for data_type in data_types.keys():
            data_types[data_type] = df[data_type_series == data_type]
    
    # 分配过滤后的数据
    zt_df = data_types['涨停']
    dt_df = data_types['跌停']
    zb_df = data_types['炸板']
    
    # 释放不再需要的数据
    del data_types
    
    # 释放原始DataFrame的内存
    del df
    
    print(f"数据过滤完成，耗时: {time() - filter_start_time:.2f}秒")
    print(f"涨停数据: {len(zt_df)}行, 跌停数据: {len(dt_df)}行, 炸板数据: {len(zb_df)}行")
    
    # 主动触发垃圾回收
    gc.collect()
    
    # 按日期分组统计
    # 记录统计开始时间
    stats_start_time = time()
    
    # 创建一个空的DataFrame用于存储统计结果
    # 使用更高效的方式初始化daily_stats - 使用set操作合并日期
    all_dates = set()
    if not zt_df.empty:
        all_dates.update(zt_df['交易日期'])
    if not dt_df.empty:
        all_dates.update(dt_df['交易日期'])
    if not zb_df.empty:
        all_dates.update(zb_df['交易日期'])
    
    # 创建日期索引
    daily_stats = pd.DataFrame(index=pd.DatetimeIndex(sorted(all_dates)))
    
    # 使用预先过滤的数据进行统计 - 一次性计算所有计数
    # 使用value_counts可能比groupby更高效
    counts_dict = {}
    
    if not zt_df.empty:
        counts_dict['涨停数'] = zt_df['交易日期'].value_counts()
    else:
        counts_dict['涨停数'] = pd.Series(0, index=daily_stats.index)
        
    if not dt_df.empty:
        counts_dict['跌停数'] = dt_df['交易日期'].value_counts()
    else:
        counts_dict['跌停数'] = pd.Series(0, index=daily_stats.index)
        
    if not zb_df.empty:
        counts_dict['炸板数'] = zb_df['交易日期'].value_counts()
    else:
        counts_dict['炸板数'] = pd.Series(0, index=daily_stats.index)
    
    # 将计数添加到daily_stats
    daily_stats = daily_stats.assign(**counts_dict)
    
    print(f"基础统计计算完成，耗时: {time() - stats_start_time:.2f}秒")
    
    # 主动触发垃圾回收
    gc.collect()
    
    # 优化热点板块统计 - 使用更高效的方法，避免循环和DataFrame复制
    # 预先计算所有日期的行业统计
    start_time = time()
    
    # 使用crosstab一次性计算所有日期和行业的组合计数
    if not zt_df.empty:
        industry_date_counts = pd.crosstab(zt_df['交易日期'], zt_df['所属行业'])
        
        # 创建热点板块字典
        hot_industries = {}
        
        # 对每个日期处理
        for date in daily_stats.index:
            if date not in industry_date_counts.index:
                hot_industries[date] = {'热点板块1': '无', '热点板块2': '无'}
                continue
                
            # 获取当前日期的行业计数
            date_counts = industry_date_counts.loc[date]
            
            # 如果没有数据，设置为无
            if date_counts.sum() == 0:
                hot_industries[date] = {'热点板块1': '无', '热点板块2': '无'}
                continue
                
            # 获取前两名行业
            top_two = date_counts.nlargest(2)
            result = [f"{industry}({count})" for industry, count in top_two.items()]
            
            # 确保始终返回两个值
            if len(result) == 0:
                hot_industries[date] = {'热点板块1': '无', '热点板块2': '无'}
            elif len(result) == 1:
                hot_industries[date] = {'热点板块1': result[0], '热点板块2': '无'}
            else:
                hot_industries[date] = {'热点板块1': result[0], '热点板块2': result[1]}
    else:
        # 如果没有涨停数据，所有日期的热点板块都是'无'
        hot_industries = {date: {'热点板块1': '无', '热点板块2': '无'} for date in daily_stats.index}
    
    # 将热点板块数据添加到daily_stats - 使用更高效的方式
    hot_industries_df = pd.DataFrame.from_dict(hot_industries, orient='index')
    daily_stats = daily_stats.join(hot_industries_df)
    
    print(f"热点板块统计耗时: {time() - start_time:.2f}秒")
    
    # 计算连板股票的最高版数 - 直接使用max函数，避免groupby操作
    if col_name:
        # 使用transform方法预计算每个日期的最高连板数
        max_tiers = zt_df.groupby('交易日期')[col_name].transform('max')
        # 创建日期到最高连板数的映射
        max_tier_dict = {}
        for date, tier in zip(zt_df['交易日期'], max_tiers):
            max_tier_dict[date] = tier
        # 将最高连板数添加到daily_stats
        daily_stats['最高连板数'] = daily_stats.index.map(lambda x: max_tier_dict.get(x, 0)).fillna(0).astype(int)
    else:
        daily_stats['最高连板数'] = 0
    
    # 优化：统计涨停梯队 - 预先计算所有日期的涨停梯队
    zt_tiers = {}
    
    if col_name:
        # 按日期分组预计算所有tier_counts
        date_tier_counts = {}
        for date in daily_stats.index:
            date_data = zt_df[zt_df['交易日期'] == date]
            if len(date_data) == 0:
                zt_tiers[date] = '无连板'
                continue
                
            # 获取最高连板数
            max_tier = int(date_data[col_name].max()) if not date_data.empty else 0
            if max_tier < 1:
                zt_tiers[date] = '无连板'
                continue
                
            # 确定需要统计的阶梯范围
            min_tier = max(1, max_tier - 4)  # 最低统计到最高连板数-4或1
            
            # 使用value_counts更高效地统计
            tier_counts = date_data[col_name].value_counts()
            
            # 格式化输出，跳过数量为0的板块
            result = []
            for tier in range(max_tier, min_tier - 1, -1):
                count = tier_counts.get(tier, 0)
                if count > 0:
                    result.append(f"{tier}{{{count}}}")
            
            zt_tiers[date] = "".join(result) if result else '无连板'
    else:
        # 如果没有连板数列，所有日期的涨停梯队都是'无连板'
        for date in daily_stats.index:
            zt_tiers[date] = '无连板'
    
    # 将涨停梯队添加到daily_stats
    daily_stats['涨停梯队'] = daily_stats.index.map(zt_tiers)
    
    # 优化：统计连板数最大的前三支股票 - 预先计算所有日期的高度板
    high_tier_stocks = {}

    if col_name and '代码' in zt_df.columns:
        for date in daily_stats.index:
            date_data = zt_df[zt_df['交易日期'] == date]
            if len(date_data) == 0:
                high_tier_stocks[date] = '无'
                continue
                
            # 获取最高连板数
            max_tier = date_data[col_name].max() if not date_data.empty else 0
            if max_tier < 1:
                high_tier_stocks[date] = '无'
                continue
                
            # 使用字典存储每个tier的股票，避免重复过滤数据
            tier_stocks = {}
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
            
            high_tier_stocks[date] = ' '.join(result_stocks) if result_stocks else '无'
    else:
        # 如果没有连板数列或代码列，所有日期的高度板都是'无'
        for date in daily_stats.index:
            high_tier_stocks[date] = '无'

    # 将高度板添加到daily_stats
    daily_stats['高度板'] = daily_stats.index.map(high_tier_stocks)
    
    # 新增：计算今日连板数（连板数>=2的股票数量）- 使用更高效的方法
    if col_name and not zt_df.empty:
        # 使用布尔索引和value_counts，可能比groupby更高效
        mask = zt_df[col_name] >= 2
        if mask.any():  # 只在有符合条件的数据时计算
            consecutive_counts = zt_df.loc[mask, '交易日期'].value_counts()
            daily_stats['今日连板数'] = consecutive_counts.reindex(daily_stats.index, fill_value=0)
        else:
            daily_stats['今日连板数'] = 0
    else:
        daily_stats['今日连板数'] = 0
    
    # 优化：计算昨日涨停数和晋级率 - 使用更高效的数据结构和算法
    # 只在涨停数据非空时进行计算
    if not zt_df.empty:
        # 获取所有日期并排序
        all_dates = sorted(zt_df['交易日期'].unique())
        
        # 使用Series.value_counts一次性计算所有日期的涨停数量
        date_to_zt_count = zt_df['交易日期'].value_counts()
        
        # 创建昨日涨停数和晋级率的Series
        yesterday_zt_counts = pd.Series(index=all_dates, dtype=float)
        promotion_rates = pd.Series(index=all_dates, dtype=float)
        tier_promotion_rates = pd.Series(index=all_dates, dtype=str)
        
        # 使用缓存优化连板级别晋级率计算
        @lru_cache(maxsize=128)
        def calculate_tier_promotion_rates(today_date, yesterday_date):
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
            
            # 计算前三个连板级别的晋级率
            result_parts = []
            count = 0
            
            for tier in range(max_tier, max_tier-3, -1):
                if tier < 2:  # 连板数至少为2
                    break
                    
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
                    count += 1
                
                # 最多统计三个级别
                if count >= 3:
                    break
            
            # 计算总体晋级率
            today_consecutive_count = sum(count for tier, count in today_tiers.items() if tier >= 2)
            yesterday_total_count = sum(yesterday_tiers.values())
            
            if yesterday_total_count > 0:
                total_rate = (today_consecutive_count / yesterday_total_count) * 100
                result_parts.append(f"总[{total_rate:.0f}%]")
            
            return "".join(result_parts)
        
        # 计算昨日涨停数和总体晋级率
        for i in range(1, len(all_dates)):
            today = all_dates[i]
            yesterday = all_dates[i-1]
            
            # 昨日涨停总数
            yesterday_count = date_to_zt_count.get(yesterday, 0)
            yesterday_zt_counts[today] = yesterday_count
            
            # 今日连板数
            if col_name:
                today_data = zt_df[zt_df['交易日期'] == today]
                today_consecutive = len(today_data[today_data[col_name] >= 2]) if not today_data.empty else 0
            else:
                today_consecutive = 0
            
            # 计算总体晋级率
            if yesterday_count > 0:
                promotion_rates[today] = (today_consecutive / yesterday_count) * 100
            else:
                promotion_rates[today] = 0
                
            # 计算各连板级别的晋级率
            tier_promotion_rates[today] = calculate_tier_promotion_rates(today, yesterday)
    
    # 将昨日涨停数和晋级率添加到daily_stats
    daily_stats['_昨日涨停数'] = yesterday_zt_counts  # 添加下划线前缀，用于内部计算
    daily_stats['晋级率'] = tier_promotion_rates  # 使用新的各级别晋级率
    
    # 记录最终处理开始时间
    final_process_start_time = time()
    
    # 统计行业分布 - 只计算一次，使用更高效的方式
    if not zt_df.empty:
        industry_counts = zt_df['所属行业'].value_counts().head(10)
    else:
        industry_counts = pd.Series(dtype='int64')
    
    # 填充缺失值为0 - 使用更高效的方式
    # 只填充数值列，避免对字符串列进行不必要的操作
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
            
            # 设置右对齐 - 批量设置单元格样式，使用列表推导式优化
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
    
    print('\n分析结果已保存到：涨停数据统计结果.xlsx')
    
    # 输出总耗时
    print(f"\n总耗时: {time() - total_start_time:.2f}秒")
    
    # 最终垃圾回收
    gc.collect()

except Exception as e:
    print(f'发生错误：{str(e)}')
    import traceback
    print(traceback.format_exc())  # 打印详细的错误堆栈信息