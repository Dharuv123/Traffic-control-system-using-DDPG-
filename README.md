# 🚦 Multi-Agent Traffic Control System using DDPG

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
How to run the model:
1. Train the model using training_main.py
2. Test the model using test_main.py
3. Check the results using plot_rewards.py
