# env/traffic_env.py
import os
import numpy as np
import gymnasium as gym
import traci
import sumolib
from generator import TrafficGeneratorIntersection

# ---------------------- Utility: SUMO Startup ---------------------- #
def _start_sumo(sumo_cfg, sumo_gui=False, seed=None):
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError("SUMO_HOME not set. Please configure your SUMO environment path.")
    sumo_binary = sumolib.checkBinary("sumo-gui") if sumo_gui else sumolib.checkBinary("sumo")
    args = [sumo_binary, "-c", sumo_cfg, "--no-step-log", "true",
            "--waiting-time-memory", "1000", "--no-warnings"]
    if seed is not None:
        args += ["--seed", str(seed)]
    traci.start(args)

# ---------------------- Single-Agent Environment ---------------------- #
class TrafficEnv(gym.Env):
    def __init__(self, sumo_cfg="config.sumocfg",
                 sumo_gui=False, max_steps=2000,
                 min_phase_duration=5, max_phase_duration=60,
                 vehicles_per_hour=1000, seed=None):
        super().__init__()
        self.sumo_cfg = sumo_cfg
        self.sumo_gui = sumo_gui
        self.max_steps = int(max_steps)
        self.min_phase_duration = int(min_phase_duration)
        self.max_phase_duration = int(max_phase_duration)
        self.vehicles_per_hour = vehicles_per_hour
        self.seed_val = seed

        # episode metrics
        self.current_step = 0
        self.entry_times = {}
        self.travel_times = []
        self.throughput = 0

        _start_sumo(self.sumo_cfg, self.sumo_gui, self.seed_val)
        self.tls_id = traci.trafficlight.getIDList()[0]
        self.phases = traci.trafficlight.getAllProgramLogics(self.tls_id)[0].phases
        self.lanes = traci.trafficlight.getControlledLanes(self.tls_id)

        self.action_size = 1
        self.observation_size = len(self.lanes) * 2
        self.observation_space = gym.spaces.Box(low=0.0, high=np.inf,
                                                shape=(self.observation_size,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0,
                                           shape=(self.action_size,), dtype=np.float32)

        self.last_phase = 0
        self.phase_steps_remaining = self.min_phase_duration

    def reset(self, *, seed=None, options=None):
        try:
            traci.close()
        except Exception:
            pass
        if seed is not None:
            self.seed_val = seed
            np.random.seed(seed)

        _start_sumo(self.sumo_cfg, self.sumo_gui, self.seed_val)

        self.tls_id = traci.trafficlight.getIDList()[0]
        self.phases = traci.trafficlight.getAllProgramLogics(self.tls_id)[0].phases
        self.lanes = traci.trafficlight.getControlledLanes(self.tls_id)

        self.current_step = 0
        self.entry_times.clear()
        self.travel_times.clear()
        self.throughput = 0
        self.last_phase = 0
        self.phase_steps_remaining = self.min_phase_duration

        return self._get_state(), {}

    def step(self, action):
        self.current_step += 1
        act = np.array(action, dtype=np.float32).reshape(-1)
        phase_count = len(self.phases)
        next_phase = int(np.floor((act[0] + 1.0) / 2.0 * (phase_count - 1)))
        next_phase = int(np.clip(next_phase, 0, phase_count - 1))

        # ✅ Scale action to phase duration
        duration = self.min_phase_duration + (act[0] + 1.0) * 0.5 * (self.max_phase_duration - self.min_phase_duration)
        duration = int(np.clip(duration, self.min_phase_duration, self.max_phase_duration))

        if self.phase_steps_remaining <= 0 or next_phase != self.last_phase:
            try:
                traci.trafficlight.setPhase(self.tls_id, next_phase)
            except traci.TraCIException:
                next_phase = int(np.clip(next_phase, 0, phase_count - 1))
                traci.trafficlight.setPhase(self.tls_id, next_phase)

            self.last_phase = next_phase
            self.phase_steps_remaining = duration

        traci.simulationStep()
        self.phase_steps_remaining = max(0, self.phase_steps_remaining - 1)

        for vid in traci.simulation.getDepartedIDList():
            self.entry_times[vid] = self.current_step

        arrived = {}
        for vid in traci.simulation.getArrivedIDList():
            if vid in self.entry_times:
                travel_time = self.current_step - self.entry_times[vid]
                self.travel_times.append(travel_time)
                arrived[vid] = travel_time
                del self.entry_times[vid]
                self.throughput += 1

        state = self._get_state()
        reward = self._compute_reward()
        terminated = traci.simulation.getMinExpectedNumber() <= 0
        truncated = self.current_step >= self.max_steps

        info = {"arrived": arrived, "throughput": self.throughput, "travel_times": self.travel_times}
        return state, reward, terminated, truncated, info

    def _get_state(self):
        queues = np.array([traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes], dtype=np.float32)
        waits = np.array([sum(traci.vehicle.getWaitingTime(vid) for vid in traci.lane.getLastStepVehicleIDs(lane))
                          for lane in self.lanes], dtype=np.float32)
        queues = queues / 10.0
        waits = waits / 100.0
        return np.concatenate([queues, waits]).astype(np.float32)

    def _compute_reward(self):
        eps = 1e-6
        total_waiting = sum(sum(traci.vehicle.getWaitingTime(vid)
                                for vid in traci.lane.getLastStepVehicleIDs(lane))
                            for lane in self.lanes)
        total_queues = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes)
        outflow = traci.simulation.getArrivedNumber()

        norm_wait = total_waiting / (len(self.lanes) + eps)
        norm_queue = total_queues / (len(self.lanes) + eps)

        reward = -0.6 * norm_wait - 0.3 * norm_queue + 0.1 * outflow
        if reward == 0:
            reward = -1.0
        return float(np.clip(reward, -50.0, 50.0))

    def close(self):
        try:
            traci.close()
        except Exception:
            pass

# ---------------------- Multi-Agent Environment ---------------------- #
class MultiAgentTrafficEnv(gym.Env):
    def __init__(self, sumo_cfg="config.sumocfg",
                 sumo_gui=False, max_steps=2000,
                 min_phase_duration=5, max_phase_duration=60,
                 vehicles_per_hour=1000, seed=None):
        super().__init__()
        self.sumo_cfg = sumo_cfg
        self.sumo_gui = sumo_gui
        self.max_steps = int(max_steps)
        self.min_phase_duration = int(min_phase_duration)
        self.max_phase_duration = int(max_phase_duration)
        self.vehicles_per_hour = vehicles_per_hour
        self.seed_val = seed

        self.current_step = 0
        self.entry_times = {}
        self.travel_times = []
        self.throughput = 0

        _start_sumo(self.sumo_cfg, self.sumo_gui, self.seed_val)
        self.tls_ids = traci.trafficlight.getIDList()
        self.agent_lanes = {tls: traci.trafficlight.getControlledLanes(tls) for tls in self.tls_ids}
        self.phases = {tls: traci.trafficlight.getAllProgramLogics(tls)[0].phases for tls in self.tls_ids}

        self.last_phases = {tls: 0 for tls in self.tls_ids}
        self.phase_steps_remaining = {tls: self.min_phase_duration for tls in self.tls_ids}

        self.observation_spaces = {
            tls: gym.spaces.Box(low=0.0, high=np.inf, shape=(len(self.agent_lanes[tls])*2,), dtype=np.float32)
            for tls in self.tls_ids
        }
        self.action_spaces = {tls: gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
                              for tls in self.tls_ids}

    def reset(self, *, seed=None, options=None):
        try:
            traci.close()
        except Exception:
            pass
        if seed is not None:
            self.seed_val = seed
            np.random.seed(seed)

        _start_sumo(self.sumo_cfg, self.sumo_gui, self.seed_val)
        self.tls_ids = traci.trafficlight.getIDList()
        self.agent_lanes = {tls: traci.trafficlight.getControlledLanes(tls) for tls in self.tls_ids}
        self.phases = {tls: traci.trafficlight.getAllProgramLogics(tls)[0].phases for tls in self.tls_ids}

        self.current_step = 0
        self.entry_times.clear()
        self.travel_times.clear()
        self.throughput = 0
        self.last_phases = {tls: 0 for tls in self.tls_ids}
        self.phase_steps_remaining = {tls: self.min_phase_duration for tls in self.tls_ids}

        return self._get_states(), {}

    def step(self, actions):
        self.current_step += 1
        for tls in self.tls_ids:
            act = np.array(actions[tls], dtype=np.float32).reshape(-1)
            phase_count = len(self.phases[tls])
            next_phase = int(np.floor((act[0]+1.0)/2.0 * (phase_count-1)))
            next_phase = int(np.clip(next_phase, 0, phase_count-1))

            # ✅ Scale action properly
            duration = self.min_phase_duration + (act[0]+1.0)*0.5*(self.max_phase_duration - self.min_phase_duration)
            duration = int(np.clip(duration, self.min_phase_duration, self.max_phase_duration))

            if self.phase_steps_remaining[tls] <= 0 or next_phase != self.last_phases[tls]:
                try:
                    traci.trafficlight.setPhase(tls, next_phase)
                except traci.TraCIException:
                    next_phase = int(np.clip(next_phase, 0, phase_count - 1))
                    traci.trafficlight.setPhase(tls, next_phase)
                self.last_phases[tls] = next_phase
                self.phase_steps_remaining[tls] = duration

            self.phase_steps_remaining[tls] = max(0, self.phase_steps_remaining[tls]-1)

        traci.simulationStep()

        for vid in traci.simulation.getDepartedIDList():
            self.entry_times[vid] = self.current_step

        arrived = {}
        for vid in traci.simulation.getArrivedIDList():
            if vid in self.entry_times:
                travel_time = self.current_step - self.entry_times[vid]
                self.travel_times.append(travel_time)
                arrived[vid] = travel_time
                del self.entry_times[vid]
                self.throughput += 1

        states = self._get_states()
        rewards = self._compute_rewards()
        terminated = traci.simulation.getMinExpectedNumber() <= 0
        truncated = self.current_step >= self.max_steps

        info = {"arrived": arrived, "throughput": self.throughput, "travel_times": self.travel_times}
        return states, rewards, terminated, truncated, info

    def _get_states(self):
        states = {}
        for tls in self.tls_ids:
            lanes = self.agent_lanes[tls]
            queues = np.array([traci.lane.getLastStepVehicleNumber(lane) for lane in lanes], dtype=np.float32)
            waits = np.array([sum(traci.vehicle.getWaitingTime(vid) for vid in traci.lane.getLastStepVehicleIDs(lane))
                              for lane in lanes], dtype=np.float32)
            queues = queues / 10.0
            waits = waits / 100.0
            states[tls] = np.concatenate([queues, waits]).astype(np.float32)
        return states

    def _compute_rewards(self):
        rewards = {}
        eps = 1e-6
        global_outflow = traci.simulation.getArrivedNumber()
        for tls in self.tls_ids:
            lanes = self.agent_lanes[tls]
            total_waiting = sum(sum(traci.vehicle.getWaitingTime(vid)
                                    for vid in traci.lane.getLastStepVehicleIDs(lane))
                                for lane in lanes)
            total_queues = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes)
            norm_wait = total_waiting / (len(lanes) + eps)
            norm_queue = total_queues / (len(lanes) + eps)
            reward = -0.6*norm_wait -0.3*norm_queue + 0.05*global_outflow
            if reward == 0:
                reward = -1.0
            rewards[tls] = float(np.clip(reward, -50.0, 50.0))
            # Debug log (optional)
            # print(f"[Reward] TLS: {tls}, Queue: {total_queues}, Wait: {total_waiting:.2f}, Reward: {rewards[tls]:.2f}")
        return rewards

    def close(self):
        try:
            traci.close()
        except Exception:
            pass
