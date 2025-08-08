import pandas as pd
import numpy as np

#读取数据并预处理
df=pd.read_csv("202503-capitalbikeshare-tripdata.csv")
df_clean=df.dropna(subset=['start_station_id', 'end_station_id'], how='any')
print(f"删去数量: {len(df) - len(df_clean)}")

#筛去异常使用记录（骑行时长异常）
"""使用3σ法筛去两侧数据"""
df_clean['started_at']=pd.to_datetime(df_clean['started_at'])
df_clean['ended_at']=pd.to_datetime(df_clean['ended_at'])
df_clean['using_duration']=(df_clean['ended_at']-df_clean['started_at']).dt.total_seconds() / 60
mean=df_clean['using_duration'].mean()
std=df_clean['using_duration'].std()
low=max(0, mean - 3 * std)
high=mean + 3 * std
df1=df_clean[(df_clean['using_duration'] >= low)&(df_clean['using_duration'] <= high)]
df1.to_csv("202503-capitalbikeshare-tripdata_1.csv", index=False)
print(f"舍去数量: {len(df_clean) - len(df1)}")

#分站点统计借还次数并形成表格
df1['start_station_id'] = df1['start_station_id'].astype(int)
df1['end_station_id'] = df1['end_station_id'].astype(int)
borrow_counts = df1.groupby('start_station_id').size().reset_index(name='borrow_count')
return_counts = df1.groupby('end_station_id').size().reset_index(name='return_count')
station_metrics = pd.merge(borrow_counts,return_counts,
    left_on='start_station_id',right_on='end_station_id',how='outer')
station_metrics['station_id'] = station_metrics['start_station_id'].fillna(station_metrics['end_station_id']).astype(int)
station_metrics = station_metrics.drop(columns=['start_station_id', 'end_station_id'])
station_metrics['borrow_count'] = station_metrics['borrow_count'].fillna(0).astype(int)
station_metrics['return_count'] = station_metrics['return_count'].fillna(0).astype(int)
station_metrics['delt_count'] = np.abs(station_metrics['borrow_count'] - station_metrics['return_count'])
station_metrics.to_csv('/mnt/站点借还量计算结果.csv', index=False)
print("\n结果已保存至：/mnt/站点借还量计算结果.csv")