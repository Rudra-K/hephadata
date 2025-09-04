import os
import numpy as np
import pandas as pd
from .model import ADM1Model
from .scenarios import SteadyStateScenario, DynamicCSVScenario

class ADM1Generator:
    def __init__(self, config_dir: str, scenario_type: str = 'dynamic'):
        print("Starting ADM1Generator:")

        params_path = os.path.join(config_dir, 'adm1_params.yaml')
        self.model = ADM1Model(params_path)

        influent_path = os.path.join(config_dir, 'influent.csv')
        if scenario_type == 'steady':
            print("Using SteadyStateScenario:")
            self.scenario = SteadyStateScenario(influent_path, self.model.state_map)
        else:
            print("Using DynamicCSVScenario:")
            self.scenario = DynamicCSVScenario(influent_path, self.model.state_map)
            
        initial_state_path = os.path.join(config_dir, 'initial_state.csv')
        self.start_state = self._load_start_state(initial_state_path)
        print("Generator initialized successfully")

    def _load_start_state(self, csv_path: str) -> np.ndarray:
        df = pd.read_csv(csv_path)
        y0 = np.zeros(len(self.model.state_map))
        for key, idx in self.model.state_map.items():
            if key in df.columns:
                y0[idx] = df[key].iloc[0]
        return y0
    
    def generate(self, duration_days: int, steps_per_day: int = 1) -> pd.DataFrame:
        t_span = (0, duration_days)
        t_eval = np.linspace(0, duration_days, duration_days * steps_per_day + 1)

        results_df = self.model.run_simulation(
            start_state=self.start_state,
            scenario=self.scenario,
            t_span=t_span,
            t_eval=t_eval
        )
        return results_df