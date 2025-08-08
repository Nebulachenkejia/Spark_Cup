import pandas as pd
import numpy as np

#读取数据并预处理
df=pd.read_csv("202503-capitalbikeshare-tripdata.csv")
#删去数据缺失的行
df_clean=df.dropna(subset=['start_station_id', 'end_station_id'], how='any')
print(f"删去数量: {len(df) - len(df_clean)}")
# 使用datetime格式
df_clean['started_at']=pd.to_datetime(df_clean['started_at'])
df_clean['ended_at']=pd.to_datetime(df_clean['ended_at'])

#筛去异常使用记录（骑行时长异常）
"""使用3σ法筛去两侧数据"""
df_clean['using_duration']=(df_clean['ended_at']-df_clean['started_at']).dt.total_seconds() / 60
mean=df_clean['using_duration'].mean()
std=df_clean['using_duration'].std()
low=max(0, mean - 3 * std)
high=mean + 3 * std
# 筛选正常骑行时间记录
df1=df_clean[(df_clean['using_duration'] >= low)&(df_clean['using_duration'] <= high)]
df1.to_csv("202503-capitalbikeshare-tripdata_1.csv", index=False)
print(f"舍去数量: {len(df_clean) - len(df1)}")