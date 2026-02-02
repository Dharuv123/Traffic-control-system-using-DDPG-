import os
import numpy as np
import tensorflow as tf
import traci
from env.traffic_env import TrafficEnv, MultiAgentTrafficEnv
from agent.ddpg_agent import DDPGAgent
from generator import TrafficGeneratorIntersection

# ---------------- Configuration ---------------- #
EPISODES = 5
MAX_STEPS = 1000
SUMO_GUI = False
VEHICLES_PER_HOUR = 1000

SINGLE_MODEL_PATH = "models/ddpg_single_final"
MULTI_MODEL_PATH = "models/ddpg_multi_final"
ROU_FILE = "episode_generated.rou.xml"

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)


# ---------------- Traffic Generation ---------------- #
def generate_route(seed=42):
    tg = TrafficGeneratorIntersection(
        max_steps=MAX_STEPS,
        vehicles_per_hour=VEHICLES_PER_HOUR,
        seed=seed
    )
    vehicles = tg.generate_vehicles()
    tg.save_rou_xml(vehicles, filename=ROU_FILE)
    print(f"✅ Generated route file with {len(vehicles)} vehicles.")


# ---------------- Single-Agent Evaluation ---------------- #
def evaluate_single():
    print("🚦 Evaluating Single-Agent Model...")
    agent = None
    all_rewards, all_awt, all_aqt = [], [], []

    for ep in range(1, EPISODES + 1):
        generate_route(seed=ep)
        env = TrafficEnv(sumo_cfg="config.sumocfg", sumo_gui=SUMO_GUI, max_steps=MAX_STEPS)
        state, _ = env.reset()

        print(f"[Ep {ep}] Vehicles at reset:", traci.vehicle.getIDList())

        if agent is None:
            state_dim = env.observation_size
            action_dim = env.action_size
            agent = DDPGAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                action_bound_low=env.min_phase_duration,
                action_bound_high=env.max_phase_duration
            )
            _ = agent.actor(tf.zeros((1, state_dim)))
            _ = agent.critic([tf.zeros((1, state_dim)), tf.zeros((1, action_dim))])
            if os.path.exists(SINGLE_MODEL_PATH + "_actor.weights.h5"):
                agent.load(SINGLE_MODEL_PATH)
                print("✅ Loaded Single-Agent model.")

        agent.reset_noise()
        total_reward = 0.0
        waiting_times = []
        queue_lengths = []

        for step in range(MAX_STEPS):
            action = agent.act(state, explore=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            state = next_state

            # AWT
            vehicle_waits = []
            for lane in env.lanes:
                vids = traci.lane.getLastStepVehicleIDs(lane)
                for v in vids:
                    vehicle_waits.append(traci.vehicle.getWaitingTime(v))
            waiting_times.append(np.mean(vehicle_waits) if vehicle_waits else 0)

            # AQT
            q = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in env.lanes) / len(env.lanes)
            queue_lengths.append(q)

            if terminated or truncated:
                break

        AWT = np.mean(waiting_times) if waiting_times else 0
        AQT = np.mean(queue_lengths) if queue_lengths else 0

        all_rewards.append(total_reward)
        all_awt.append(AWT)
        all_aqt.append(AQT)

        print(f"[Single Ep {ep}] Reward: {total_reward:.2f} | AWT: {AWT:.2f} | AQT: {AQT:.2f}")
        env.close()

    np.save(os.path.join(results_dir, "single_rewards.npy"), np.array(all_rewards))
    np.save(os.path.join(results_dir, "single_awt.npy"), np.array(all_awt))
    np.save(os.path.join(results_dir, "single_aqt.npy"), np.array(all_aqt))
    print("✅ Single-Agent evaluation complete.\n")


# ---------------- Multi-Agent Evaluation ---------------- #
def evaluate_multi():
    print("🚦 Evaluating Multi-Agent Model...")
    agent = None
    all_rewards, all_awt, all_aqt = [], [], []

    for ep in range(1, EPISODES + 1):
        generate_route(seed=ep + 100)
        env = MultiAgentTrafficEnv(sumo_cfg="config.sumocfg", sumo_gui=SUMO_GUI, max_steps=MAX_STEPS)
        states, _ = env.reset()

        print(f"[Ep {ep}] Vehicles at reset:", traci.vehicle.getIDList())

        if agent is None:
            tls_id = list(env.agent_lanes.keys())[0]
            state_dim = len(env.agent_lanes[tls_id]) * 2
            agent = DDPGAgent(
                state_dim=state_dim,
                action_dim=1,
                action_bound_low=-1.0,
                action_bound_high=1.0
            )
            _ = agent.actor(tf.zeros((1, state_dim)))
            _ = agent.critic([tf.zeros((1, state_dim)), tf.zeros((1, 1))])
            if os.path.exists(MULTI_MODEL_PATH + "_actor.weights.h5"):
                agent.load(MULTI_MODEL_PATH)
                print("✅ Loaded Multi-Agent model.")

        agent.reset_noise()
        total_reward = 0.0
        waiting_times = []
        queue_lengths = []

        for step in range(MAX_STEPS):
            actions = agent.act(states, explore=False)
            next_states, rewards, terminated, truncated, info = env.step(actions)
            total_reward += sum(rewards.values())
            states = next_states

            # AWT
            vehicle_waits = []
            for tls in env.tls_ids:
                for lane in env.agent_lanes[tls]:
                    vids = traci.lane.getLastStepVehicleIDs(lane)
                    for v in vids:
                        vehicle_waits.append(traci.vehicle.getWaitingTime(v))
            waiting_times.append(np.mean(vehicle_waits) if vehicle_waits else 0)

            # AQT
            total_q = 0
            count = 0
            for tls in env.tls_ids:
                for lane in env.agent_lanes[tls]:
                    total_q += traci.lane.getLastStepVehicleNumber(lane)
                    count += 1
            queue_lengths.append(total_q / count if count else 0)

            if terminated or truncated:
                break

        AWT = np.mean(waiting_times) if waiting_times else 0
        AQT = np.mean(queue_lengths) if queue_lengths else 0

        all_rewards.append(total_reward)
        all_awt.append(AWT)
        all_aqt.append(AQT)

        print(f"[Multi Ep {ep}] Reward: {total_reward:.2f} | AWT: {AWT:.2f} | AQT: {AQT:.2f}")
        env.close()

    np.save(os.path.join(results_dir, "multi_rewards.npy"), np.array(all_rewards))
    np.save(os.path.join(results_dir, "multi_awt.npy"), np.array(all_awt))
    np.save(os.path.join(results_dir, "multi_aqt.npy"), np.array(all_aqt))
    print("✅ Multi-Agent evaluation complete.\n")


# ---------------- Main ---------------- #
if __name__ == "__main__":
    evaluate_single()
    evaluate_multi()
    print("🎯 Evaluation finished. Results saved in /results folder.")
