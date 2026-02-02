# training_main.py
import os
import numpy as np
import random
from env.traffic_env import MultiAgentTrafficEnv
from agent.ddpg_agent import DDPGAgent
from generator import TrafficGeneratorIntersection

# ---------------------- Config ---------------------- #
EPISODES = 200
MAX_STEPS = 1000
SUMO_GUI = False
MIN_PHASE = 12
MAX_PHASE = 60

MULTI_MODEL_PATH = "models/ddpg_multi_final"
ROU_FILE = "intersection_generated.rou.xml"

# Dynamic route generation parameters
SAVE_EVERY = 10
SEED_BASE = 42
VEHICLE_MIN = 1500
VEHICLE_MAX = 2500

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ---------------------- Dynamic Route Generator ---------------------- #
def generate_dynamic_routes(filename=ROU_FILE, vehicles_per_hour=2000, max_steps=3600, seed=None):
    tg = TrafficGeneratorIntersection(max_steps=max_steps,
                                      vehicles_per_hour=vehicles_per_hour,
                                      seed=seed)
    vehicles = tg.generate_vehicles()
    tg.save_rou_xml(vehicles, filename=filename)


# ---------------------- Training Function ---------------------- #
def train_agent(env, agent, episodes, max_steps, model_path):
    episode_rewards = []

    try:
        for ep in range(1, episodes + 1):

            seed = SEED_BASE + ep
            vehicles_per_hour = random.randint(VEHICLE_MIN, VEHICLE_MAX)

            # Route file per episode
            generate_dynamic_routes(
                filename=ROU_FILE,
                vehicles_per_hour=vehicles_per_hour,
                max_steps=max_steps,
                seed=seed
            )

            states, _ = env.reset(seed=seed)
            agent.reset_noise()
            total_reward = 0.0

            for step in range(max_steps):

                actions = agent.act(states, explore=True)
                next_states, rewards, terminated, truncated, _ = env.step(actions)
                done = terminated or truncated

                for tls_id, s in states.items():
                    r = rewards.get(tls_id, 0.0)
                    ns = next_states[tls_id]
                    a = actions[tls_id]
                    agent.remember(s, a, r, ns, done)

                agent.train()
                total_reward += sum(rewards.values())
                states = next_states

                if done:
                    break

            episode_rewards.append(total_reward)
            avg_reward = np.mean(episode_rewards[-10:])

            print(f"[Multi Ep {ep:03d}/{episodes}] "
                  f"Reward: {total_reward:.2f} | Avg(10): {avg_reward:.2f} "
                  f"| Vehicles/hr: {vehicles_per_hour}")

            if ep % SAVE_EVERY == 0:
                agent.save(model_path)
                np.save(model_path + "_rewards.npy", np.array(episode_rewards))

        agent.save(model_path)
        np.save(model_path + "_rewards.npy", np.array(episode_rewards))

        print(f"✅ Finished multi-agent training → saved to {model_path}")

    finally:
        try:
            env.close()
        except:
            pass

    return episode_rewards


# ---------------------- MAIN ---------------------- #
if __name__ == "__main__":

    print("▶ Starting Multi-Agent (Shared Policy) Training Only...")
    generate_dynamic_routes(filename=ROU_FILE,
                            vehicles_per_hour=2000,
                            max_steps=MAX_STEPS,
                            seed=SEED_BASE)

    multi_env = MultiAgentTrafficEnv(
        sumo_cfg="config.sumocfg",
        sumo_gui=SUMO_GUI,
        max_steps=MAX_STEPS,
        min_phase_duration=MIN_PHASE,
        max_phase_duration=MAX_PHASE
    )

    first_tls = list(multi_env.observation_spaces.keys())[0]
    state_dim = multi_env.observation_spaces[first_tls].shape[0]

    multi_agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=1,
        action_bound_low=-1.0,
        action_bound_high=1.0
    )

    train_agent(
        multi_env,
        multi_agent,
        EPISODES,
        MAX_STEPS,
        MULTI_MODEL_PATH
    )

    print("🎯 Multi-Agent Training Completed Successfully.")
