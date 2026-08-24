# 🚦 Multi-Agent Traffic Control System using DDPG

**Tech Stack:** Python, TensorFlow, SUMO, TraCI

## 📌 Project Overview
Traffic congestion is a major problem in urban areas due to fixed and non-adaptive traffic signal systems. Traditional traffic lights operate on predefined timings and cannot respond effectively to real-time traffic variations.

This project proposes an **intelligent, decentralized traffic signal control system** using **Multi-Agent Reinforcement Learning (MARL)** with the **Deep Deterministic Policy Gradient (DDPG)** algorithm. Each traffic intersection is modeled as an autonomous learning agent that dynamically adjusts signal timings based on real-time traffic conditions.

The system is implemented and tested using the **SUMO traffic simulator** integrated with Python via the **TraCI interface**.

---

## 🎯 Objectives
- Reduce vehicle waiting time at intersections
- Minimize queue length and congestion
- Enable real-time adaptive traffic signal control
- Eliminate dependency on centralized traffic controllers
- Improve overall traffic throughput

---

## 🧠 Key Concepts Used
- Reinforcement Learning (RL)
- Multi-Agent Reinforcement Learning (MARL)
- Deep Deterministic Policy Gradient (DDPG)
- Actor–Critic Architecture
- Continuous Action Space
- Decentralized Control
- Traffic Simulation using SUMO

---

## 🏗 System Architecture

1. **SUMO Traffic Simulator**
   Simulates real-world traffic conditions with vehicles, lanes, and signalized intersections.

2. **TraCI Interface**
   Acts as a bridge between SUMO and Python code, enabling real-time data exchange and control.

3. **Agent Observation Layer**
   Extracts traffic state information such as queue length and waiting time.

4. **DDPG Learning Agent**
   - Actor Network: Decides optimal green signal duration
   - Critic Network: Evaluates action quality
   - Replay Buffer: Stores past experiences
   - Target Networks: Stabilize learning

5. **Performance Evaluation**
   Measures waiting time, queue length, and traffic throughput.

---

## 🔁 Working Methodology

1. Traffic state is observed from SUMO using TraCI
2. Each agent selects an action (green time duration) using the Actor network
3. Action is applied to the traffic signal
4. Environment updates traffic flow
5. Reward is calculated based on congestion reduction
6. Experience is stored and used to train the agent
7. The process repeats over multiple episodes

---

## 📊 Results

- Achieved up to **40% reduction in vehicle waiting time** compared to fixed-timing baseline signals
- Reduced average queue length across simulated intersections
- Decentralized agents learned effective signal-control policies without a centralized controller

*(Add any additional numbers you have — number of episodes trained, convergence behavior, throughput improvement, etc. — to strengthen this section further.)*

---

## 🚀 Setup

### Prerequisites
- Python 3.8+
- [SUMO](https://sumo.dlr.de/docs/Downloads.php) installed and added to your system PATH
- TensorFlow and other dependencies

### Installation
```bash
git clone https://github.com/Dharuv123/YOUR-REPO-NAME.git
cd marl-traffic-control-ddpg
pip install -r requirements.txt
```

---

## ▶️ How to Run

1. **Train the model:**
   ```bash
   python training_main.py
   ```

2. **Test the trained model:**
   ```bash
   python test_main.py
   ```

3. **Visualize results:**
   ```bash
   python plot_rewards.py
   ```

---

## 🔮 Future Improvements

- Extend to a centralized-training, decentralized-execution (CTDE) MARL setup for better agent coordination
- Compare DDPG performance against other algorithms (e.g., PPO, MADDPG)
- Scale to larger, real-world road networks
- Add a live dashboard for real-time training/evaluation metrics

---

## 👤 Author

**Dharuvkumar Bhansali**
[LinkedIn](https://www.linkedin.com/in/dharuvkumar-bhansali-b1319425b/) · [GitHub](https://github.com/Dharuv123)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
