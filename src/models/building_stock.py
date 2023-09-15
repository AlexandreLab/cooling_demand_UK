from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.common import schema, sim_param
from src.models import thermal_model
from src.data import source


@dataclass
class BuildingStock:
  dwelling_parameters: dict[str, float]

  # Create models
  def create_dwelling(self) -> thermal_model.ThermalModel:
    R = self.dwelling_parameters['R']
    C = self.dwelling_parameters['C']

    dwelling = thermal_model.ThermalModel(R=R, C=C)
    dwelling.g_t = self.dwelling_parameters['g_t']
    dwelling.floor_area = self.dwelling_parameters['floor_area']
    dwelling.volume_rooms = sum(
        self.dwelling_parameters['volume_rooms'].values())
    dwelling.air_change_rate = self.dwelling_parameters['air_flow_rate']
    return dwelling

  def run_simulation(self, dwelling: thermal_model.ThermalModel,
                     input_rcmodel_dataf: pd.DataFrame) -> pd.DataFrame:
    dwelling.load_model_data(input_rcmodel_dataf)
    rcmodel_dataf = dwelling.run_model()
    rcmodel_dataf.index.name = schema.DataSchema.TIME_HOURS
    return rcmodel_dataf

  def get_simulation_data(self, weather_data: pd.DataFrame) -> pd.DataFrame:
    data_source = source.SimulationData(weather_data)
    rcmodel_dataf = data_source.create_CIBSE_based_simulation_data()
    return rcmodel_dataf