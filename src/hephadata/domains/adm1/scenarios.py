import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class Scenario(ABC):
    @abstractmethod
    def get_influent_state(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def get_flow_rate(self, t:float) -> float:
        pass

class SteadyStateScenario(Scenario):
    def __init__(self, influent_csv_path: str, state_map: dict):
        influent_df = pd.read_csv(influent_csv_path)

        self._flow_rate = influent_df['q_ad'].iloc[0]

        self._influent_state = np.zeros(len(state_map))
        for key, idx in state_map.items():
            if key in influent_df.columns:
                self._influent_state[idx] = influent_df[key].iloc[0]

    def get_influent_state(self, t: float) -> np.ndarray:
        return self._influent_state
    
    def get_flow_rate(self, t: float) -> float:
        return self._flow_rate
    
class DynamicCSVScenario(Scenario):
    def __init__(self, influent_csv_path: str, state_map: dict):
        self._df = pd.read_csv(influent_csv_path)

        self._influent_states = {}

        for index, row in self._df.iterrows():
            time = row['time']
            state_vector = np.zeros(len(state_map))
            for key, idx in state_map.items():
                if key in self._df.columns:
                    state_vector[idx] = row[key]
            self._influent_states[time] = state_vector

    def get_influent_state(self, t: float) -> np.ndarray:
        relevent_time = self._df.loc[self._df['time'] <= t, 'time'].max()
        return self._influent_states[relevent_time]
    
    def get_flow_rate(self, t: float) -> float:
        flow_rate = self._df.loc[self._df['time'] <= t, 'q_ad'].iloc[-1]
        return float(flow_rate)