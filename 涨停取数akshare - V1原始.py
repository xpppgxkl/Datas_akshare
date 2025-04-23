import akshare as ak
import pandas as pd
import time
import datetime
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import ConnectionError, Timeout

# 配置基本日志 - 减少不必要的输出以提高性能
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('涨停数据')

def fetch_stock_data(date_str, api_func, retry_count=2, sleep_time=0.2):
    """高效获取单个API数据，自动重试"""
    for attempt in range(retry_count + 1):
        try:
            data = api_func(date=date_str)
            # 提前清理数据以减少内存占用
            if data is not None and not data.empty:
                # 清理涨停统计列
                if '涨停统计' in data.columns:
                    # 直接使用批量替换，不需要apply
                    data['涨停统计'] = (data['涨停统计'].astype(str)
                                     .str.replace(r'日|月|年', '', regex=True)
                                     .str.replace('/', '//', regex=False))
                return data
            else:
                return None
        except (ConnectionError, Timeout):
            if attempt < retry_count:
                time.sleep(sleep_time * (attempt + 1))  # 递增休眠
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
    """主函数 - 高效获取和处理数据"""
    start_time = time.time()
    
    try:
        # 获取交易日历
        trade_dates = ak.tool_trade_date_hist_sina()
        # 转换为datetime并筛选
        trade_dates['trade_date'] = pd.to_datetime(trade_dates['trade_date'])
        recent_dates = trade_dates[trade_dates['trade_date'] <= pd.Timestamp.today()].tail(30)['trade_date']
        
        # 确定最佳线程数 - 针对高性能CPU优化
        date_count = len(recent_dates)
        # 对于16核心32线程的CPU，可以使用更多线程
        # 但API限制需要考虑，建议最大值在12-16之间
        worker_count = min(16, max(date_count // 2, 1))
        
        logger.info(f"开始获取{date_count}个交易日数据，使用{worker_count}个并行工作线程 (优化for 16核CPU)")
        
        # 将日期分成小批次，减少线程创建和管理开销
        batch_size = max(1, date_count // worker_count)
        date_batches = [recent_dates[i:i+batch_size] for i in range(0, date_count, batch_size)]
        
        # 存储所有结果
        all_results = []
        
        # 使用线程池并行处理批次
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
        
        # 合并所有数据类型的数据
        dataframes = []
        
        # 高效地收集每种类型的数据
        for data_type in ['zt', 'dt', 'zb']:
            type_dfs = []
            for result in all_results:
                if data_type in result and result[data_type] is not None:
                    type_dfs.append(result[data_type])
            
            if type_dfs:
                dataframes.append(pd.concat(type_dfs, ignore_index=True))
        
        # 如果没有数据则退出
        if not dataframes:
            logger.warning("未获取到有效数据")
            return
        
        # 合并所有类型数据
        final_df = pd.concat(dataframes, ignore_index=True)
        
        # 优化内存使用 - 转换数据类型
        optimize_cols = {
            '代码': 'str', 
            '名称': 'str',
            '涨停统计': 'str',
            '交易日期': 'str',
            '数据类型': 'category',  # 使用category类型节省内存
        }
        
        for col, dtype in optimize_cols.items():
            if col in final_df.columns:
                final_df[col] = final_df[col].astype(dtype)
        
        # 排序数据
        final_df.sort_values(['交易日期', '数据类型'], ascending=[False, True], inplace=True)
        
        # 保存文件 - 修改为CSV格式
        output_dir = "W:\\PyFiles\\Datas_akshare"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"market_data_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        # 直接保存为CSV格式
        logger.info("正在将数据保存为CSV格式...")
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"数据已成功保存为CSV文件: {output_file}")
        
        # 输出统计信息
        logger.info(f"数据处理完成: {len(final_df)}行 x {len(final_df.columns)}列")
        logger.info(f"总执行时间: {time.time() - start_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    main()