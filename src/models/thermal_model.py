from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.common import schema, sim_param


@dataclass
class ThermalModel:
  """Create a thermal model of a building."""
  R: float = 0  # K/kW
  C: float = 0  # kJ/K
  floor_area: float = 50
  volume_rooms: float = 0  #m3
  air_change_rate = 0.33  #air changes per second
  cooling_design_temperature: float = sim_param.TARGET_INDOOR_AIR_TEMP  #28
  heating_design_temperature: float = sim_param.TARGET_INDOOR_AIR_TEMP  #-5
  target_indoor_air_temperature: float = sim_param.TARGET_INDOOR_AIR_TEMP
  initial_heating_output: float = 0
  model_data = pd.DataFrame()
  g_t = 0.76  # Transmittance factors for glazing 0.85 (single glazed) to 0.57 (triple glazed), 0.76 double glazed Table 6b p216.
  frame_factor = 0.7  #0.7-0.8 in Table 6c p216
  summer_solar_access = 0.9  # Summer solar access factor: 0.9 (Average value) Table 6d p216

  @property
  def exp_factor(self) -> float:
    return np.exp(-sim_param.TIMESTEP_SIMULATION / (self.R * self.C))

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

  @property
  def max_cooling_output(self) -> float:
    if self.cooling_design_temperature > self.target_indoor_air_temperature:
      cooling_output = -1 / self.R * (self.cooling_design_temperature -
                                      self.target_indoor_air_temperature)
    else:
      cooling_output = 0
    return cooling_output

  @property
  def max_heating_output(self) -> float:
    if self.heating_design_temperature < self.target_indoor_air_temperature:
      heating_output = 1 / self.R * (self.target_indoor_air_temperature -
                                     self.heating_design_temperature)
    else:
      heating_output = 0
    return heating_output

  def load_model_data(self, dataf: pd.DataFrame) -> None:
    self.model_data = dataf.copy()
    self.estimate_solar_gains()
    gains_cols = [
        schema.DataSchema.TOTALGAINS, schema.DataSchema.SOLARGAINS,
        schema.DataSchema.APPLIANCESGAINS, schema.DataSchema.OCCUPANCYGAINS
    ]
    total_gains = self.model_data[gains_cols].sum(axis=1).values
    self.model_data[schema.DataSchema.TOTALGAINS] = total_gains

  def estimate_solar_gains(self) -> pd.DataFrame:
    """Estimate the solar gains of the building. Return a copy of the simulation data."""

    def solar_gains(solar_flux: float) -> float:
      correction_factor = 1
      A = 0.25 * self.floor_area  #Opening areas (including windows, roof windows, rooflights and doors), max 25% of total floor area
      return correction_factor * 0.9 * A * solar_flux * self.g_t * self.frame_factor * self.summer_solar_access / 1000

    self.model_data[schema.DataSchema.SOLARGAINS] = self.model_data[
        schema.DataSchema.SOLARRADIATION].apply(lambda x: solar_gains(x))
    return self.model_data

  def calculate_appliances_internal_heat_gains(self, month: int, nb_days: int):
    """"Calculate the internal heat gains of appliances in kW"""
    monthly_energy_use = self.annual_appliance_energy_use * (
        1 + 0.157 * math.cos(2 * math.pi *
                             (month - 1.78) / 12)) * nb_days / 365
    return monthly_energy_use / (24 * nb_days)

  def calculate_occupancy_heat_gains(self) -> float:
    """"Metabolic internal heat gains from occupancy in kW (see table 5, SAP 2012)"""
    return self.number_occupants * 60 / 1000

  def calc_ventilation_losses(self, iat: float, oat: float) -> float:
    q = 1.225  #kg/m3
    cp = 1005  # J/kg.K
    mass_flow_rate = q * self.air_change_rate * self.volume_rooms
    return (iat - oat) * mass_flow_rate * cp / 1000  #kW

  def run_model(self) -> pd.DataFrame:
    #https://www.sciencedirect.com/science/article/pii/S0360544213005525
    dataf = self.model_data.copy()
    time_index: npt.NDArray[np.int64] = dataf.index.to_numpy()
    indoor_air_temperatures: npt.NDArray[np.float64] = dataf[
        schema.DataSchema.IAT].to_numpy()
    outdoor_air_temperatures: npt.NDArray[np.float64] = dataf[
        schema.DataSchema.OAT].to_numpy()
    heating_system_outputs: npt.NDArray[np.float64] = dataf[
        schema.DataSchema.HEATINGOUTPUT].to_numpy()
    heat_gains: npt.NDArray[np.float64] = dataf[
        schema.DataSchema.TOTALGAINS].to_numpy()
    ventilation_losses: npt.NDArray[np.float64] = np.zeros(len(dataf))

    for t in np.arange(1, len(time_index)):
      ventilation_losses[t] = self.calc_ventilation_losses(
          indoor_air_temperatures[t - 1], outdoor_air_temperatures[t - 1])
      temp_gains = (heat_gains[t] - ventilation_losses[t])
      temp_iat = indoor_air_temperatures[t - 1]
      temp_oat = outdoor_air_temperatures[t - 1]

      indoor_air_temperatures[t] = self.estimated_iat_without_heating(
          temp_iat, temp_oat, temp_gains)
      estimated_heating_required = self.estimated_heating_system_output(
          indoor_air_temperatures[t])

      indoor_air_temperatures[t] = indoor_air_temperatures[t] + self.R * (
          1 - self.exp_factor) * estimated_heating_required
      heating_system_outputs[t] = estimated_heating_required

    dataf[schema.DataSchema.IAT] = indoor_air_temperatures
    dataf[schema.DataSchema.HEATINGOUTPUT] = heating_system_outputs
    dataf[schema.DataSchema.VENTILATION] = ventilation_losses
    return dataf.copy()

  def estimated_iat_without_heating(self, iat: float, oat: float,
                                    gains: float) -> float:
    return iat * self.exp_factor + (1 - self.exp_factor) * oat + self.R * (
        1 - self.exp_factor) * gains

  def estimated_heating_system_output(self,
                                      estimated_iat: float,
                                      heating_season_flag: int = 0) -> float:
    """Estimated heating output required to get to the target temperature. By default, heating season flag is set to 0."""
    #-(outdoor_air_temperatures[t]-indoor_air_temperatures[t-1])/self.R-solar_gains[t] to keep temperature close to target temperature
    estimated_heating_required = (self.target_indoor_air_temperature -
                                  estimated_iat) / (self.R *
                                                    (1 - self.exp_factor))
    estimated_heating_required = self.cap_estimated_heating_system_output(
        estimated_heating_required, heating_season_flag)

    #calculate the expected indoor air temperature
    estimated_iat = estimated_iat + self.R * (
        1 - self.exp_factor) * estimated_heating_required
    if estimated_iat > sim_param.MAX_INDOOR_AIR_TEMP or estimated_iat > self.target_indoor_air_temperature:
      if not heating_season_flag:  #cooling season
        #Cooling required to avoid going above threshold
        estimated_heating_required = self.max_cooling_output
      else:
        estimated_heating_required = 0

    elif estimated_iat < sim_param.MIN_INDOOR_AIR_TEMP or estimated_iat < self.target_indoor_air_temperature:
      #Heating required to avoid going above threshold
      if heating_season_flag:  #heating season
        estimated_heating_required = self.max_heating_output
      else:
        estimated_heating_required = 0
    return estimated_heating_required

  def cap_estimated_heating_system_output(self, estimated_heating: float,
                                          heating_season_flag: int) -> float:
    if heating_season_flag:  #heating season no cooling allowed
      if estimated_heating < 0:
        estimated_heating = 0
      if estimated_heating > self.max_heating_output:
        estimated_heating = self.max_heating_output
    else:  #cooling season, no heating allowed
      if estimated_heating > 0:
        estimated_heating = 0
      if estimated_heating < self.max_cooling_output:
        estimated_heating = self.max_cooling_output
    return estimated_heating
