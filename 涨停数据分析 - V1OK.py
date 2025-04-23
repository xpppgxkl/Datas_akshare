import pandas as pd
import numpy as np

try:
    # 读取Excel文件
    df = pd.read_excel(r"W:\PyFiles\Datas_akshare\market_data_analysis.xlsx")
    
    # 检查必要的列是否存在
    required_columns = ['交易日期', '数据类型', '所属行业']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f'缺少必要的列：{col}')
    
    # 确保日期列为datetime类型
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    
    # 按日期分组统计
    daily_stats = pd.DataFrame()
    
    # 统计每天的涨停数、跌停数和炸板数
    daily_stats['涨停数'] = df[df['数据类型'] == '涨停'].groupby('交易日期').size()
    daily_stats['跌停数'] = df[df['数据类型'] == '跌停'].groupby('交易日期').size()
    daily_stats['炸板数'] = df[df['数据类型'] == '炸板'].groupby('交易日期').size()
    
    # 统计每日热点板块（涨停数量最多的前两个行业）
    def get_hot_industries(group):
        if len(group) == 0:
            return pd.Series({'热点板块1': '无', '热点板块2': '无'})
        industry_counts = group['所属行业'].value_counts()
        if len(industry_counts) == 0:
            return pd.Series({'热点板块1': '无', '热点板块2': '无'})
        # 获取前两名行业
        top_two = industry_counts.head(2)
        result = [f"{industry}({count})" for industry, count in top_two.items()]
        # 确保始终返回两个值
        if len(result) == 0:
            return pd.Series({'热点板块1': '无', '热点板块2': '无'})
        elif len(result) == 1:
            return pd.Series({'热点板块1': result[0], '热点板块2': '无'})
        else:
            return pd.Series({'热点板块1': result[0], '热点板块2': result[1]})
    
    # 只对涨停数据统计热点板块
    zt_df = df[df['数据类型'] == '涨停']
    daily_hot_industries = zt_df.groupby('交易日期').apply(get_hot_industries)
    daily_stats['热点板块1'] = daily_hot_industries['热点板块1']
    daily_stats['热点板块2'] = daily_hot_industries['热点板块2']
    
    # 计算连板股票的最高版数
    def get_max_consecutive(group):
        if len(group) == 0:
            return 0
        # 确保连续涨停天数列存在
        if '连续涨停天数' in group.columns:
            return group['连续涨停天数'].max()
        elif '连板数' in group.columns:
            return group['连板数'].max()
        return 0
    
    # 只对涨停数据计算最高连板数
    zt_df = df[df['数据类型'] == '涨停']
    daily_stats['最高连板数'] = zt_df.groupby('交易日期').apply(get_max_consecutive)
    
    # 新增：统计涨停梯队
    def get_zt_tiers(group):
        if len(group) == 0:
            return '无连板'
        
        # 确定使用哪个列作为连板数
        if '连续涨停天数' in group.columns:
            col_name = '连续涨停天数'
        elif '连板数' in group.columns:
            col_name = '连板数'
        else:
            return '无连板数据'
        
        # 获取最高连板数
        max_tier = int(group[col_name].max())
        if max_tier < 1:
            return '无连板'
        
        # 确定需要统计的阶梯范围
        min_tier = max(1, max_tier - 4)  # 最低统计到最高连板数-4或1
        
        # 统计每个阶梯的股票数量
        tier_counts = {}
        for tier in range(max_tier, min_tier - 1, -1):
            count = sum(group[col_name] == tier)
            tier_counts[tier] = count
        
        # 格式化输出，跳过数量为0的板块
        result = []
        for tier in range(max_tier, min_tier - 1, -1):
            if tier in tier_counts and tier_counts[tier] > 0:  # 只显示数量大于0的板块
                result.append(f"{tier}{{{tier_counts[tier]}}}")
        
        return "".join(result)
    
    # 添加涨停梯队列
    daily_stats['涨停梯队'] = zt_df.groupby('交易日期').apply(get_zt_tiers)
    
    # 新增：统计连板数最大的前三支股票
    def get_top_consecutive_stocks(group):
        if len(group) == 0:
            return '无'
        
        # 确定使用哪个列作为连板数
        if '连续涨停天数' in group.columns:
            col_name = '连续涨停天数'
        elif '连板数' in group.columns:
            col_name = '连板数'
        else:
            return '无连板数据'
        
        # 确保股票代码列存在
        if '代码' not in group.columns:
            return '无股票代码数据'
        
        # 按连板数降序排序，取前三
        top_stocks = group.sort_values(by=col_name, ascending=False).head(3)
        
        # 格式化输出：股票代码[连板数]
        result = []
        for _, row in top_stocks.iterrows():
            stock_code = str(row['代码']).zfill(6)  # 确保股票代码为6位，不足前面补0
            tier = int(row[col_name])
            result.append(f"{stock_code}[{tier}]")
        
        return ';'.join(result) if result else '无'
    
    # 添加高度板列
    daily_stats['高度板'] = zt_df.groupby('交易日期').apply(get_top_consecutive_stocks)
    
    # 统计行业分布
    industry_counts = df['所属行业'].value_counts()
    top_industries = industry_counts.head(10)
    
    # 填充缺失值为0
    daily_stats = daily_stats.fillna(0)
    
    # 按日期降序排序
    daily_stats = daily_stats.sort_index(ascending=False)
    
    # 将索引（日期）转换为字符串格式，只保留年月日
    daily_stats.index = daily_stats.index.strftime('%Y-%m-%d')
    
    # 打印结果
    print('\n每日统计数据：')
    print(daily_stats)
    
    print('\n出现最多的行业（前10）：')
    print(top_industries)
    
    # 保存结果到Excel，并设置涨停梯队列右对齐
    output_file = '涨停数据统计结果.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        daily_stats.to_excel(writer, sheet_name='Sheet1')
        
        # 获取工作簿和工作表
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        
        # 找到需要右对齐的列的索引（列号）
        zt_tier_col_idx = daily_stats.columns.get_loc('涨停梯队') + 2  # +2是因为Excel列从1开始，且第一列是索引
        gd_tier_col_idx = daily_stats.columns.get_loc('高度板') + 2
        
        # 设置涨停梯队列和高度板列右对齐
        from openpyxl.styles import Alignment
        for row in range(2, len(daily_stats) + 2):  # 从第2行开始（跳过标题行）
            for col_idx in [zt_tier_col_idx, gd_tier_col_idx]:
                cell = worksheet.cell(row=row, column=col_idx)
                cell.alignment = Alignment(horizontal='right')
    
    print('\n分析结果已保存到：涨停数据统计结果.xlsx')

except Exception as e:
    print(f'发生错误：{str(e)}')