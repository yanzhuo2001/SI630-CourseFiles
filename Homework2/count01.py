import pandas as pd

# 读取CSV文件
df = pd.read_csv('sentiment.dev.csv')

# 统计label列中0和1的数量
count_0 = (df['label'] == 0).sum()
count_1 = (df['label'] == 1).sum()

print(f"Count of 0: {count_0}")
print(f"Count of 1: {count_1}")
