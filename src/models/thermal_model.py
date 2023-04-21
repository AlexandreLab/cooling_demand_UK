from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models import dwelling_functions

from ..common import schema


@dataclass
class ThermalModel:
  """Create a thermal model of a building."""
  R: float = 0    # K/kW
  C: float = 0    # kJ/K
  floor_area: float = 50
  outdoor_air_temperature: float = 0
  initial_indoor_air_temperature: float = 21
  limit_indoor_air_temperature: float = 0
  cooling_design_temperature: float = 28
  heating_design_temperature: float = -5
  target_indoor_air_temperature: float = 21
  initial_heating_output: float = 0
  model_data = pd.DataFrame()
  max_cooling_output: float = 0.
  max_heating_output: float = 0.
  g_t = 0.76    # Transmittance factors for glazing 0.85 (single glazed) to 0.57 (triple glazed), 0.76 double glazed Table 6b p216.
  frame_factor = 0.7    #0.7-0.8 in Table 6c p216
  summer_solar_access = 0.9    # Summer solar access factor: 0.9 (Average value) Table 6d p216

  def init_parameters(self):
    self.set_max_cooling_output()
    self.set_max_heating_output()
    self.initial_heating_output = self.get_initial_heating_output()
    self.max_heating_output = round(
        1 / self.R *
        (self.target_indoor_air_temperature - self.heating_design_temperature),
        3,
    )
    return self

  @property
  def annual_appliance_energy_use(self):
    return 207.8 * (self.floor_area * self.number_occupants)**0.4714

  @property
  def number_occupants(self) -> float:
    """Function to calculate the number of occupants based on the floor area (see SAP 2012 table 1b)"""
    nb_occupants: float = 0.
    if self.floor_area <= 13.9:
      nb_occupants = 1
    else:
      nb_occupants = 1 + 1.76 * (1 - math.exp(
          -0.000349 *
          (self.floor_area - 13.9)**2)) + 0.0013 * (self.floor_area - 13.9)
    return nb_occupants

  def set_max_cooling_output(self) -> None:
    self.max_cooling_output = -1 / self.R * (self.cooling_design_temperature -
                                             self.target_indoor_air_temperature)

  def set_max_heating_output(self) -> None:
    self.max_heating_output = 1 / self.R * (self.target_indoor_air_temperature -
                                            self.heating_design_temperature)

  def get_initial_heating_output(self) -> float:
    delta_T = self.initial_indoor_air_temperature - self.outdoor_air_temperature
    init_P_out = 1 / self.R * delta_T
    if init_P_out < 0:
      init_P_out = 0
    if init_P_out > self.max_heating_output:
      init_P_out = self.max_heating_output
    return round(init_P_out, 3)

  def get_duration_service(self, current_heating_output):

    return dwelling_functions.calculate_duration_service(
        self.initial_indoor_air_temperature,
        self.outdoor_air_temperature,
        self.limit_indoor_air_temperature,
        current_heating_output,
        self.max_heating_output,
        self.R,
        self.C,
    )

  def get_max_temperature(self, current_heating_output):
    return dwelling_functions.calculate_max_temperature(
        self.outdoor_air_temperature, current_heating_output, self.R)

  def get_heating_output(self, duration):
    #     print(f'Dwelling with R={R}, C={C}, T_in={T_in}, T_out={T_out}, duration={duration} sec')
    B = 1 - 1 / (self.R * self.C)
    P_out = ((self.limit_indoor_air_temperature -
              self.initial_indoor_air_temperature * B**duration -
              self.outdoor_air_temperature / (self.R * self.C) *
              (1 - B**duration) / (1 - B)) * self.C * (1 - B) /
             (1 - B**duration))
    #     if P_out<0:
    #         P_out = 0
    return P_out

  def load_model_data(self, dataf: pd.DataFrame) -> None:
    self.model_data = dataf

  def run_model(self, dataf: pd.DataFrame) -> pd.DataFrame:
    #https://www.sciencedirect.com/science/article/pii/S0360544213005525
    time_index = dataf.index.values
    indoor_air_temperatures = dataf[schema.DataSchema.IAT].values
    outdoor_air_temperatures = dataf[schema.DataSchema.OAT].values
    heating_system_outputs = dataf[schema.DataSchema.HEATINGOUTPUT].values
    heat_gains = dataf[schema.DataSchema.TOTALGAINS].values

    for t in np.arange(1, len(time_index)):
      timestep = (time_index[t] - time_index[t - 1])
      exp_factor = np.exp(-timestep / (self.R * self.C))
      indoor_air_temperatures[
          t] = indoor_air_temperatures[t - 1] * exp_factor + (
              1 - exp_factor) * outdoor_air_temperatures[t] + self.R * (
                  1 - exp_factor) * (heat_gains[t] + heating_system_outputs[t])

    dataf[schema.DataSchema.IAT] = indoor_air_temperatures
    return dataf.copy()

  def estimate_heating_demand(
      self,
      dataf: pd.DataFrame,
      min_indoor_air_temperature: float = 19,
      max_indoor_air_temperature: float = 24) -> pd.DataFrame:
    """Estimate the heating/cooling output required to maintain indoor air temperature between limits."""
    #https://www.sciencedirect.com/science/article/pii/S0360544213005525
    time_index = dataf.index.values
    indoor_air_temperatures = dataf[schema.DataSchema.IAT].values
    outdoor_air_temperatures = dataf[schema.DataSchema.OAT].values
    heating_system_outputs = dataf[schema.DataSchema.HEATINGOUTPUT].values
    heat_gains = dataf[schema.DataSchema.TOTALGAINS].values
    heating_season_flag = dataf[schema.DataSchema.HEATINGSEASON].values

    heating_system_outputs[0] = 0
    for t in np.arange(1, len(time_index)):
      #Assume that the heating output remain constant compared to previous timestep.
      timestep = (time_index[t] - time_index[t - 1])
      exp_factor = np.exp(-timestep / (self.R * self.C))

      estimated_iat_without_heating = indoor_air_temperatures[
          t - 1] * exp_factor + (1 - exp_factor) * outdoor_air_temperatures[
              t] + self.R * (1 - exp_factor) * (heat_gains[t])

      estimated_heating_required = (
          self.target_indoor_air_temperature - estimated_iat_without_heating
      ) / (
          self.R * (1 - exp_factor)
      )    #-(outdoor_air_temperatures[t]-indoor_air_temperatures[t-1])/self.R-solar_gains[t]
      if heating_season_flag[t]:    #heating season no cooling allowed
        if estimated_heating_required < 0:
          estimated_heating_required = 0
        if estimated_heating_required > self.max_heating_output:
          estimated_heating_required = self.max_heating_output
      else:    #cooling season, no heating allowed
        if estimated_heating_required > 0:
          estimated_heating_required = 0
        if estimated_heating_required < self.max_cooling_output:
          estimated_heating_required = self.max_cooling_output

      #calculate the expected indoor air temperature
      estimated_iat = estimated_iat_without_heating + self.R * (
          1 - exp_factor) * estimated_heating_required

      if estimated_iat > max_indoor_air_temperature or estimated_iat > self.target_indoor_air_temperature:
        if not heating_season_flag[t]:    #cooling season
          #Cooling required to avoid going above threshold
          estimated_heating_required = self.max_cooling_output
        else:
          estimated_heating_required = 0

      elif estimated_iat < min_indoor_air_temperature or estimated_iat < self.target_indoor_air_temperature:
        #Heating required to avoid going above threshold
        if heating_season_flag[t]:    #heating season
          estimated_heating_required = self.max_heating_output
        else:
          estimated_heating_required = 0
      indoor_air_temperatures[t] = estimated_iat_without_heating + self.R * (
          1 - exp_factor) * estimated_heating_required
      heating_system_outputs[t] = estimated_heating_required

    dataf[schema.DataSchema.IAT] = indoor_air_temperatures
    dataf[schema.DataSchema.HEATINGOUTPUT] = heating_system_outputs
    return dataf.copy()
