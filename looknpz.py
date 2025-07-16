import numpy as np

# 1) 读文件
data = np.load("data/events_20250309170420868.npz")  # 改成你的 npz 路径
print("keys:", data.files)        # ['t', 'x', 'y', 'p']

# 2) 取出各列
t = data["t"]     # float64, 秒
x = data["x"]     # uint16
y = data["y"]     # uint16
p = data["p"]     # int8 (+1/-1)

print("total events:", t.size)
print("时间范围:", t.min(), "→", t.max())
print(x.min(), x.max())
print(y.min(), y.max())

# 随机看前 5 个事件
for i in range(5):
    print(f"{t[i]:.6f}s  (x={x[i]}, y={y[i]}, p={p[i]})")

