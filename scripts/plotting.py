import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("runs/fineweb-llama-132m/metrics.csv")
df = df.drop_duplicates(subset="step", keep="last")  # in case the run was resumed

val = val = df.dropna(subset=["val_loss"])
plt.figure(figsize=(8, 5))
plt.plot(
    df["step"],
    df["train_loss"].rolling(50).mean(),
    label="train (rolling mean)",
    alpha=0.7,
)
plt.plot(val["step"], val["val_loss"], "o-", label="val")
plt.ylim(top=4.5)  # zoom in; remove to see the full drop from ~11
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"final val loss: {val['val_loss'].iloc[-1]:.4f}")
print(f"best val loss:  {val['val_loss'].min():.4f}")
print(f"median tok/s:   {df['tokens_per_sec'].iloc[5:].median():,.0f}")
