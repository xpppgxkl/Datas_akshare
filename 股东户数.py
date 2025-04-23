import akshare as ak

stock_zh_a_gdhs_df = ak.stock_zh_a_gdhs(symbol='20230930')
print(stock_zh_a_gdhs_df)

stock_zh_a_gdhs_df.to_excel('W:\PyFiles\数据分析akshare\股东户数.xlsx')