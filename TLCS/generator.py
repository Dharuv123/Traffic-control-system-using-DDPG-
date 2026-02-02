import os
import numpy as np


class TrafficGeneratorIntersection:
    """
    Generates dynamic traffic route files (.rou.xml) for SUMO intersections.
    Supports initial vehicles at t=0.
    """

    def __init__(self, max_steps=3600, vehicles_per_hour=2000, seed=None,
                 initial_vehicles_per_route=0):
        self.max_steps = int(max_steps)
        self.vehicles_per_hour = int(vehicles_per_hour)
        self.initial_vehicles_per_route = int(initial_vehicles_per_route)

        # Seed
        self.seed = seed if seed is not None else np.random.randint(0, 10000)
        np.random.seed(self.seed)

    # -------------------------------------------------------------
    def generate_vehicles(self):
        """
        Generate vehicle departure times using Weibull distribution.
        Also adds initial vehicles at step 0.
        """
        n_vehicles = int(self.vehicles_per_hour * self.max_steps / 3600)

        # Weibull traffic generation
        timings = np.random.weibull(a=2.0, size=n_vehicles)
        timings = np.sort(timings)

        # Normalize
        t_min, t_max = timings.min(), timings.max()
        car_gen_steps = ((timings - t_min) / (t_max - t_min)) * self.max_steps
        car_gen_steps = np.rint(car_gen_steps).astype(int)

        # Add initial vehicles at t = 0
        if self.initial_vehicles_per_route > 0:
            initial_cars = np.zeros(self.initial_vehicles_per_route, dtype=int)
            car_gen_steps = np.concatenate([initial_cars, car_gen_steps])

        return car_gen_steps

    # -------------------------------------------------------------
    def save_rou_xml(self, car_gen_steps, filename="intersection/intersection_generated.rou.xml"):
        """ Save SUMO route file. """

        # Create directory
        dirpath = os.path.dirname(filename)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

        with open(filename, "w") as f:
            f.write("<routes>\n")
            f.write(
                '    <vType id="car" accel="1.0" decel="4.5" length="5.0" '
                'minGap="2.5" maxSpeed="25.0" sigma="0.5" color="1,0,0"/>\n'
            )

            # 12 possible routes
            routes = {
                "W_E": "W_in E_out", "W_N": "W_in N_out", "W_S": "W_in S_out",
                "E_W": "E_in W_out", "E_N": "E_in N_out", "E_S": "E_in S_out",
                "N_S": "N_in S_out", "N_W": "N_in W_out", "N_E": "N_in E_out",
                "S_N": "S_in N_out", "S_W": "S_in W_out", "S_E": "S_in E_out"
            }

            for rid, edges in routes.items():
                f.write(f'    <route id="{rid}" edges="{edges}"/>\n')

            route_ids = list(routes.keys())

            for i, step in enumerate(car_gen_steps):
                route = np.random.choice(route_ids)
                f.write(
                    f'    <vehicle id="{route}_{i}" type="car" route="{route}" '
                    f'depart="{int(step)}" departLane="random" departSpeed="10"/>\n'
                )

            f.write("</routes>\n")

        
