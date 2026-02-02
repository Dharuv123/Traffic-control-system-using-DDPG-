import os
import matplotlib.pyplot as plt
import numpy as np

# ----------------- Load Results ----------------- #
if not os.path.exists("results"):
    raise FileNotFoundError("❌ No results folder found. Run evaluation first!")

single_rewards = np.load("results/single_rewards.npy")
multi_rewards = np.load("results/multi_rewards.npy")
single_awt = np.load("results/single_awt.npy")   # Avg Waiting Time
multi_awt = np.load("results/multi_awt.npy")
single_att = np.load("results/single_att.npy")   # Avg Travel Time  <<< NEW
multi_att = np.load("results/multi_att.npy")
single_aqt = np.load("results/single_aqt.npy")   # Avg Queue Time   <<< NEW
multi_aqt = np.load("results/multi_aqt.npy")

episodes = np.arange(1, len(single_rewards) + 1)

# ----------------- Print Summary ----------------- #
print("\n📊 === Evaluation Summary ===")
print(f"Single-Agent Avg Reward: {np.mean(single_rewards):.2f}")
print(f"Multi-Agent  Avg Reward: {np.mean(multi_rewards):.2f}")
print(f"Single-Agent Avg AWT:    {np.mean(single_awt):.2f}")
print(f"Multi-Agent  Avg AWT:    {np.mean(multi_awt):.2f}")
print(f"Single-Agent Avg ATT:    {np.mean(single_att):.2f}")
print(f"Multi-Agent  Avg ATT:    {np.mean(multi_att):.2f}")
print(f"Single-Agent Avg AQT:    {np.mean(single_aqt):.2f}")
print(f"Multi-Agent  Avg AQT:    {np.mean(multi_aqt):.2f}")
print(f"Best Single-Agent AWT:   {np.min(single_awt):.2f}")
print(f"Best Multi-Agent AWT:    {np.min(multi_awt):.2f}\n")

# Ensure folder exists
os.makedirs("results", exist_ok=True)


# ----------------- Plot 1: Rewards ----------------- #
plt.figure(figsize=(8, 5))
plt.plot(episodes, single_rewards, marker='o', label="Single-Agent")
plt.plot(episodes, multi_rewards, marker='x', label="Multi-Agent")
plt.title("Total Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("results/rewards_plot.png")
print("📌 Saved: results/rewards_plot.png")
plt.show()


# ----------------- Plot 2: Average Waiting Time ----------------- #
plt.figure(figsize=(8, 5))
plt.plot(episodes, single_awt, marker='o', label="Single-Agent")
plt.plot(episodes, multi_awt, marker='x', label="Multi-Agent")
plt.title("Average Waiting Time per Episode (AWT)")
plt.xlabel("Episode")
plt.ylabel("Avg Waiting Time (sec)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("results/awt_plot.png")
print("📌 Saved: results/awt_plot.png")
plt.show()


# ----------------- Plot 3: Average Travel Time ----------------- #
plt.figure(figsize=(8, 5))
plt.plot(episodes, single_att, marker='o', label="Single-Agent")
plt.plot(episodes, multi_att, marker='x', label="Multi-Agent")
plt.title("Average Travel Time per Episode (ATT)")
plt.xlabel("Episode")
plt.ylabel("Avg Travel Time (sec)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("results/att_plot.png")
print("📌 Saved: results/att_plot.png")
plt.show()


# ----------------- Plot 4: Average Queue Time ----------------- #
plt.figure(figsize=(8, 5))
plt.plot(episodes, single_aqt, marker='o', label="Single-Agent")
plt.plot(episodes, multi_aqt, marker='x', label="Multi-Agent")
plt.title("Average Queue Time per Episode (AQT)")
plt.xlabel("Episode")
plt.ylabel("Avg Queue Time (sec)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("results/aqt_plot.png")
print("📌 Saved: results/aqt_plot.png")
plt.show()
