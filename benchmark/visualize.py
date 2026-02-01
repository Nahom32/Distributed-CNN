import subprocess
import matplotlib.pyplot as plt
import re
import time
import os

# --- CONFIGURATION ---
EXECUTABLE = "./../build/cnn_dist"

EPOCHS = 5
PROCESSOR_COUNTS = [1, 2, 4]


def parse_metrics(output):
    """Extracts Epoch, Loss, and Accuracy from stdout"""
    epochs = []
    losses = []
    accuracies = []

    for line in output.splitlines():
        if line.startswith("METRICS"):
            # Format: METRICS,epoch,loss,accuracy
            parts = line.split(",")
            epochs.append(int(parts[1]))
            losses.append(float(parts[2]))
            accuracies.append(float(parts[3]))

    return epochs, losses, accuracies


def parse_time(output):
    """Extracts total execution time"""
    for line in output.splitlines():
        if line.startswith("TIME_TOTAL"):
            return float(line.split(",")[1])
    return None


def run_experiment(n_procs, n_epochs):
    print(f"Running with {n_procs} processes for {n_epochs} epochs...")
    cmd = ["mpirun", "-n", str(n_procs), EXECUTABLE, str(n_epochs)]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error running C code:")
        print(result.stderr)
        return None, None

    return result.stdout


def plot_training_curves(epochs, losses, accuracies):
    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color=color)
    ax1.plot(epochs, losses, color=color, marker="o", label="Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Test Accuracy", color=color)
    ax2.plot(
        epochs, accuracies, color=color, marker="s", linestyle="--", label="Accuracy"
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Distributed CNN Training Progress")
    fig.tight_layout()
    plt.grid(True)

    filename = "training_metrics.png"
    plt.savefig(filename)
    print(f"Saved training plot to {filename}")
    plt.show()


def plot_speedup(proc_counts, times):
    base_time = times[0]
    speedups = [base_time / t for t in times]
    ideal = proc_counts

    plt.figure()
    plt.plot(proc_counts, speedups, "o-", label="Actual Speedup", linewidth=2)
    plt.plot(proc_counts, ideal, "r--", label="Ideal Linear Speedup", alpha=0.5)

    plt.xlabel("Number of MPI Processes")
    plt.ylabel("Speedup Factor")
    plt.title("Strong Scaling on M4 Chip")
    plt.legend()
    plt.grid(True)
    plt.xticks(proc_counts)

    filename = "speedup_analysis.png"
    plt.savefig(filename)
    print(f"Saved speedup plot to {filename}")
    plt.show()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(EXECUTABLE):
        print(f"Error: Could not find executable at {EXECUTABLE}. Did you run 'make'?")
        exit(1)

    # 1. VISUALIZE TRAINING ACCURACY
    # Run with maximum processors to get the best training curve
    print("--- Phase 1: Training Visualization ---")
    output = run_experiment(n_procs=4, n_epochs=EPOCHS)
    if output:
        ep, loss, acc = parse_metrics(output)
        plot_training_curves(ep, loss, acc)

    # 2. VISUALIZE SPEEDUP
    # Run short epochs just to measure speed
    print("\n--- Phase 2: Speedup Analysis ---")
    times = []
    valid_procs = []

    for p in PROCESSOR_COUNTS:
        # Run for only 1 epoch to save time during benchmarking
        out = run_experiment(n_procs=p, n_epochs=1)
        if out:
            t = parse_time(out)
            if t:
                times.append(t)
                valid_procs.append(p)
                print(f"  {p} Processors -> {t:.2f} seconds")

    if len(times) > 0:
        plot_speedup(valid_procs, times)
