import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/wine.csv")

x = df["alcohol"].to_numpy()

y = df["quality"].to_numpy()

a, b = np.polyfit(x, y, 1)

print("傾き =", a)
print("切片 =", b)

y_pred = a * x + b

plt.scatter(x, y)
plt.plot(x, y_pred, color="red")

plt.xlabel("alcohol")
plt.ylabel("quality")


