import akshare as ak
import pandas as pd
import time
import datetime
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import ConnectionError, Timeout
import csv  # 添加这个导入

# 配置基本日志 - 减少不必要的输出以提高性能
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('涨停数据')

def fetch_stock_data(date_str, api_func, retry_count=2, sleep_time=0.2):
    """高效获取单个API数据，确保文本格式和前导零保留"""
    for attempt in range(retry_count + 1):
        try:
            data = api_func(date=date_str)
            if data is not None and not data.empty:
                # 确保代码列作为文本处理，保留前导零
                if '代码' in data.columns:
                    data['代码'] = data['代码'].astype(str).str.zfill(6)
                
                # 清理涨停统计列
                if '涨停统计' in data.columns:
                    # 先处理NaN值，将它们替换为空字符串
                    data['涨停统计'] = data['涨停统计'].fillna('')
                    # 只处理非空的值
                    mask = data['涨停统计'] != ''
                    if mask.any():
                        data.loc[mask, '涨停统计'] = (data.loc[mask, '涨停统计'].astype(str)
                                                  .str.replace(r'日|月|年', '', regex=True)
                                                  .str.replace('/', '|', regex=False))
                return data
            else:
                return None
        except (ConnectionError, Timeout):
            if attempt < retry_count:
                time.sleep(sleep_time * (attempt + 1))
        except Exception as e:
            logger.debug(f"API调用异常: {str(e)}")
            return None
    return None

def process_date_data_batch(dates_batch):
    """批量处理日期数据，减少函数调用开销"""
    results = []
    
    # 定义API映射，便于统一处理
    api_types = {
        'zt': {'func': ak.stock_zt_pool_em, 'name': '涨停'},
        'dt': {'func': ak.stock_zt_pool_dtgc_em, 'name': '跌停'},
        'zb': {'func': ak.stock_zt_pool_zbgc_em, 'name': '炸板'}
    }
    
    for date in dates_batch:
        date_str = date.strftime('%Y%m%d')
        date_result = {}
        
        for api_key, api_info in api_types.items():
            # 获取数据
            df = fetch_stock_data(date_str, api_info['func'])
            
            # 如果获取到数据，添加必要的列
            if df is not None and not df.empty:
                df['交易日期'] = date.strftime('%Y-%m-%d')
                df['数据类型'] = api_info['name']
                date_result[api_key] = df
        
        # 只有在至少有一个API返回数据时才添加结果
        if date_result:
            results.append(date_result)
            
    return results

def main():
    """主函数 - 确保数据唯一性和文本格式"""
    start_time = time.time()
    
    try:
        # 获取交易日历
        trade_dates = ak.tool_trade_date_hist_sina()
        # 转换为datetime并筛选
        trade_dates['trade_date'] = pd.to_datetime(trade_dates['trade_date'])
        recent_dates = trade_dates[trade_dates['trade_date'] <= pd.Timestamp.today()].tail(30)['trade_date']
        
        # 确定最佳线程数 - 针对高性能CPU优化
        date_count = len(recent_dates)
        worker_count = min(16, max(date_count // 2, 1))
        
        logger.info(f"开始获取{date_count}个交易日数据，使用{worker_count}个并行工作线程")
        
        # 获取新数据
        batch_size = max(1, date_count // worker_count)
        date_batches = [recent_dates[i:i+batch_size] for i in range(0, date_count, batch_size)]
        
        all_results = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_batch = {
                executor.submit(process_date_data_batch, batch): i 
                for i, batch in enumerate(date_batches)
            }
            
            for future in as_completed(future_to_batch):
                batch_results = future.result()
                if batch_results:
                    all_results.extend(batch_results)
                    logger.info(f"完成批次{future_to_batch[future]+1}/{len(date_batches)}处理")
        
        # 处理新数据
        dataframes = []
        for data_type in ['zt', 'dt', 'zb']:
            type_dfs = []
            for result in all_results:
                if data_type in result and result[data_type] is not None:
                    type_dfs.append(result[data_type])
            
            if type_dfs:
                dataframes.append(pd.concat(type_dfs, ignore_index=True))
        
        if not dataframes:
            logger.warning("未获取到新的有效数据")
            return
        
        # 合并新获取的所有类型数据
        new_data = pd.concat(dataframes, ignore_index=True)
        
        # 确保代码列为文本格式，保留前导零
        if '代码' in new_data.columns:
            new_data['代码'] = new_data['代码'].astype(str).str.zfill(6)
        
        # 确保新数据无重复，使用交易日期和名称判断
        if '交易日期' in new_data.columns and '名称' in new_data.columns:
            before_dedup = len(new_data)
            new_data = new_data.drop_duplicates(subset=['交易日期', '名称'], keep='first')
            after_dedup = len(new_data)
            if before_dedup > after_dedup:
                logger.info(f"从新数据中移除了 {before_dedup - after_dedup} 条重复记录")
        
        # 为数据添加唯一标识符用于比较
        if '交易日期' in new_data.columns and '名称' in new_data.columns:
            # 创建唯一标识：日期+名称
            new_data['标识'] = new_data['交易日期'].astype(str) + '_' + new_data['名称'].astype(str)
        
        # 固定的本地数据文件路径
        local_data_file = "W:\\PyFiles\\Datas_akshare\\market_data_analysis.csv"
        
        # 读取本地存储的历史数据
        existing_data = pd.DataFrame()
        if os.path.exists(local_data_file):
            logger.info(f"读取本地数据文件: {local_data_file}")
            
            try:
                # 指定所有列为字符串类型，防止数值转换
                existing_data = pd.read_csv(local_data_file, encoding='utf-8-sig', dtype=str)
                logger.info(f"已读取本地数据: {len(existing_data)}行")
                
                # 确保代码列格式正确
                if '代码' in existing_data.columns:
                    existing_data['代码'] = existing_data['代码'].astype(str).str.zfill(6)
                
            except Exception as e:
                logger.warning(f"读取本地数据文件失败: {e}，将创建新文件")
        else:
            logger.info(f"本地数据文件不存在，将创建新文件: {local_data_file}")
        
        # 数据合并逻辑修改
        if not existing_data.empty:
            # 确保交易日期和代码都存在
            if '交易日期' in new_data.columns and '代码' in new_data.columns and '名称' in new_data.columns:
                # 为新旧数据创建唯一标识（使用交易日期+代码作为唯一标识符）
                new_data['标识'] = new_data['交易日期'].astype(str) + '_' + new_data['代码'].astype(str)
                
                if '交易日期' in existing_data.columns and '代码' in existing_data.columns:
                    existing_data['标识'] = existing_data['交易日期'].astype(str) + '_' + existing_data['代码'].astype(str)
                    
                    # 找出新数据中的标识
                    new_identifiers = set(new_data['标识'])
                    
                    # 保留本地数据中不在新数据中的记录
                    keep_old = existing_data[~existing_data['标识'].isin(new_identifiers)]
                    
                    # 合并数据
                    if not keep_old.empty:
                        # 删除临时标识列
                        keep_old = keep_old.drop('标识', axis=1)
                        new_data = new_data.drop('标识', axis=1)
                        
                        # 合并新旧数据
                        merged_data = pd.concat([keep_old, new_data], ignore_index=True)
                        logger.info(f"数据合并完成: 保留{len(keep_old)}行旧数据，添加{len(new_data)}行新数据")
                        logger.info(f"对于相同日期下的个股，已以新数据为准")
                    else:
                        # 如果所有旧数据都被替换，直接使用新数据
                        merged_data = new_data.drop('标识', axis=1)
                        logger.info(f"所有旧数据已更新为新数据")
                else:
                    merged_data = new_data.drop('标识', axis=1) if '标识' in new_data.columns else new_data
                    logger.warning("本地数据格式不兼容，使用新数据替代")
            else:
                merged_data = pd.concat([existing_data, new_data], ignore_index=True)
                logger.warning("新数据缺少必要字段，直接追加到本地数据")
        else:
            merged_data = new_data
            if '标识' in merged_data.columns:
                merged_data = merged_data.drop('标识', axis=1)
            logger.info("未找到本地数据，创建新文件")
        
        # 确保所有列都保持为字符串格式，特别是代码列
        for col in merged_data.columns:
            # 处理NaN值，将它们替换为空字符串
            merged_data[col] = merged_data[col].fillna('')
            
            # 处理字符串中的"nan"值
            merged_data[col] = merged_data[col].replace('nan', '')
            
            # 特别处理代码列，确保前导零
            if col == '代码':
                # 只处理非空值
                mask = merged_data[col] != ''
                if mask.any():
                    merged_data.loc[mask, col] = merged_data.loc[mask, col].str.zfill(6)
        
        # 再次去重以确保最终数据唯一性
        if '交易日期' in merged_data.columns and '代码' in merged_data.columns and '数据类型' in merged_data.columns:
            before_final_dedup = len(merged_data)
            merged_data = merged_data.drop_duplicates(subset=['交易日期', '代码', '数据类型'], keep='first')
            after_final_dedup = len(merged_data)
            if before_final_dedup > after_final_dedup:
                logger.info(f"从最终数据中移除了 {before_final_dedup - after_final_dedup} 条重复记录")
        
        # 排序数据
        if '交易日期' in merged_data.columns and '数据类型' in merged_data.columns:
            merged_data.sort_values(['交易日期', '数据类型'], ascending=[False, True], inplace=True)
        
        # 保存合并后的数据到固定路径
        logger.info("正在保存增量更新后的数据...")
        
        # 直接保存数据，不创建备份
        try:
            with open(local_data_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)  # 引用所有字段
                # 写入表头
                writer.writerow(merged_data.columns)
                # 写入数据行，将"nan"替换为空字符串
                for _, row in merged_data.iterrows():
                    # 处理每一行中的nan值
                    cleaned_row = ['' if (val == 'nan' or val == 'None' or val is None) else val for val in row]
                    writer.writerow(cleaned_row)
            logger.info(f"数据已以纯文本格式保存至: {local_data_file}")
        except Exception as e:
            logger.error(f"保存文件时发生错误: {e}")
        
        # 输出统计信息
        logger.info(f"最终数据: {len(merged_data)}行 x {len(merged_data.columns)}列")
        logger.info(f"总执行时间: {time.time() - start_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    main()