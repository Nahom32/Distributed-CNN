import matplotlib.pyplot as plt
import numpy as np

# --- YOUR DATA ---
epochs = [1, 2, 3, 4, 5]

# Serial Data (from Screenshot 2)
serial_loss = [0.5432, 0.3061, 0.2737, 0.2452, 0.2174]
serial_acc = [90.52, 91.50, 92.46, 93.21, 93.97]
serial_time = 163.60

# Distributed Data (from Screenshot 1)
dist_loss = [1.0758, 0.3996, 0.3483, 0.3262, 0.3118]
dist_acc = [88.19, 90.30, 90.90, 91.29, 91.55]
dist_time = 45.97

# --- PLOT 1: Speedup Bar Chart ---
plt.figure(figsize=(8, 6))
times = [serial_time, dist_time]
labels = ["Serial (1 Core)", "Distributed (4 Cores)"]
colors = ["#ff9999", "#66b3ff"]

bars = plt.bar(labels, times, color=colors, edgecolor="black")

# Add text on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 2,
        f"{yval:.2f}s",
        ha="center",
        va="bottom",
        fontsize=12,
    )

# Add Speedup Arrow
plt.annotate(
    f"Speedup: {serial_time / dist_time:.2f}x",
    xy=(1, dist_time),
    xytext=(0.5, (serial_time + dist_time) / 2),
    arrowprops=dict(arrowstyle="->", lw=2, color="black"),
    fontsize=12,
    fontweight="bold",
    ha="center",
)

plt.ylabel("Total Execution Time (seconds)")
plt.title("Performance Speedup: Serial vs. Distributed")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("plot_speedup.png")
plt.show()

# --- PLOT 2: Accuracy Comparison ---
plt.figure(figsize=(10, 6))
plt.plot(epochs, serial_acc, "o--", label="Serial Accuracy", color="red", linewidth=2)
plt.plot(
    epochs, dist_acc, "s-", label="Distributed Accuracy", color="blue", linewidth=2
)

plt.xlabel("Epochs")
plt.ylabel("Test Accuracy (%)")
plt.title("Convergence Comparison: Serial vs Distributed")
plt.legend()
plt.grid(True)
plt.xticks(epochs)
plt.savefig("plot_accuracy.png")
plt.show()
