from __future__ import annotations

import calendar
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import icecream as ic
import numpy as np
import numpy.typing as npt
import pandas as pd
from dotenv import load_dotenv

from common import schema, sim_param

load_dotenv()  # take environment variables from .env.


@dataclass
class EquipmentGainsProfile:
  normalised_profile: pd.DataFrame = field(default_factory=pd.DataFrame)

  def __post_init__(self):
    if len(self.normalised_profile) == 0:
      str_path: str = os.getenv('PATH_EQUIPMENT_GAINS_PROFILE')
      sheet_name: str = os.getenv('SHEET_EQUIPMENT_GAINS_PROFILE')
      ic.ic(str_path, sheet_name)
      equipment_gains_df = pd.read_excel(Path(str_path),
                                         sheet_name=sheet_name,
                                         index_col=0)
      self.normalised_profile = equipment_gains_df.loc[
          'Normalised profile', :].to_frame()
    sum_profile = self.normalised_profile['Normalised profile'].sum().round(4)
    ic.ic(sum_profile)
    if sum_profile != 1:
      ic.ic(
          'The sum of the normalised profile is not equal to 1, please check')

  def create_normalised_profile_from_datetimeindex(
      self, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    profile_dict = dict(self.normalised_profile.reset_index().values)
    profile_df = pd.DataFrame(index=date_index)
    profile_df[schema.DataSchema.APPLIANCESGAINS] = [
        profile_dict[x + 1] for x in profile_df.index.hour
    ]
    return profile_df

  def create_appliances_internal_heat_gains_profile(
      self, annual_appliance_energy_use: float,
      date_index: pd.DatetimeIndex) -> npt.NDArray:
    """"Calculate the internal heat gains of appliances in kW, L11, section L2 in SAP 2012.
    January = 1 to December = 12"""
    profile_df = self.create_normalised_profile_from_datetimeindex(date_index)
    year = date_index.year[0]
    for month in date_index.month.unique():

      nb_days = calendar.monthrange(year, month)[1]
      monthly_energy_use = annual_appliance_energy_use * (
          1 + 0.157 * math.cos(2 * math.pi *
                               (month - 1.78) / 12)) * nb_days / 365
      energy_per_day = monthly_energy_use / nb_days
      # ic.ic(energy_per_day)
      filt = profile_df.index.month == month
      temp_values = profile_df.loc[
          filt, schema.DataSchema.APPLIANCESGAINS].values * energy_per_day
      profile_df.loc[filt, schema.DataSchema.APPLIANCESGAINS] = temp_values
    return profile_df[schema.DataSchema.APPLIANCESGAINS].values


@dataclass
class ThermalModel:
  """Create a thermal model of a building."""
  R: float = 0  # K/kW
  C: float = 0  # kJ/K
  floor_area: float = 50
  air_change_rate: float = 0.15  #air changes per hour when windows are closed (infiltration rate)
  air_change_rate_window_open: float = 4  #air changes per hour when windows are open
  cooling_design_temperature: float = sim_param.COOLING_DESIGN_AIR_TEMP
  heating_design_temperature: float = sim_param.HEATING_TARGET_INDOOR_AIR_TEMP
  heating_target_indoor_air_temperature: float = sim_param.HEATING_TARGET_INDOOR_AIR_TEMP
  cooling_target_indoor_air_temperature: float = sim_param.COOLING_TARGET_INDOOR_AIR_TEMP
  initial_heating_output: float = 0
  model_data = pd.DataFrame()
  g_t: float = 0.76  # Transmittance factors for glazing 0.85 (single glazed) to 0.57 (triple glazed), 0.76 double glazed Table 6b p216.
  frame_factor: float = 0.7  #0.7-0.8 in Table 6c p216
  summer_solar_access: float = 0.9  # Summer solar access factor: 0.9 (Average value) Table 6d p216
  opening_window_iat_limit: float = 22  # windows are opened if iat above 22C
  opening_window_oat_limit: float = 26  # windows can be opened if oat below 26C
  equipment_profile: EquipmentGainsProfile = EquipmentGainsProfile()
  overwrite_volume_rooms: float = 0

  @property
  def volume_rooms(self) -> float:
    # https://www.designsindetail.com/articles/whats-the-uks-standard-ceiling-height-for-houses-extensions-and-loft-conversions#:~:text=Ceiling%20height%20standards,75%25%20of%20the%20floor%20area.
    if self.overwrite_volume_rooms == 0:
      return self.floor_area * 2.3
    else:
      return self.overwrite_volume_rooms

  @property
  def exp_factor(self) -> float:
    return np.exp(-sim_param.TIMESTEP_SIMULATION_INT / (self.R * self.C))

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
    if self.cooling_design_temperature > self.cooling_target_indoor_air_temperature:
      cooling_output = -1 / self.R * (
          self.cooling_design_temperature -
          self.cooling_target_indoor_air_temperature)
    else:
      cooling_output = 0
    return cooling_output

  @property
  def max_heating_output(self) -> float:
    if self.heating_design_temperature < self.heating_target_indoor_air_temperature:
      heating_output = 1 / self.R * (self.heating_target_indoor_air_temperature
                                     - self.heating_design_temperature)
    else:
      heating_output = 0
    return heating_output

  def load_model_data(self, dataf: pd.DataFrame) -> None:
    # ic.ic(self)
    self.model_data = dataf.copy()
    self.estimate_solar_gains()
    self.model_data[
        schema.DataSchema.
        APPLIANCESGAINS] = self.equipment_profile.create_appliances_internal_heat_gains_profile(
            self.annual_appliance_energy_use, dataf.index)
    gains_cols = [
        schema.DataSchema.TOTALGAINS,
        schema.DataSchema.SOLARGAINS,
        schema.DataSchema.OCCUPANCYGAINS,
        schema.DataSchema.APPLIANCESGAINS,
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
    """"Calculate the internal heat gains of appliances in kW, L11, section L2 in SAP 2012"""
    monthly_energy_use = self.annual_appliance_energy_use * (
        1 + 0.157 * math.cos(2 * math.pi *
                             (month - 1.78) / 12)) * nb_days / 365
    return monthly_energy_use / (24 * 3600 /
                                 sim_param.TIMESTEP_SIMULATION_INT * nb_days)

  # def create_equipment_gains_profile(self)->pd.Series:
  # profile_df =
  # sim_dataf = pd.merge(self.model_data, profile_df, how='left', left_on='Hour', right_on=profile_df.index-1)
  # pass

  def calculate_occupancy_heat_gains(self) -> float:
    """"Metabolic internal heat gains from occupancy in kW (see table 5, SAP 2012)"""
    return self.number_occupants * 60 / 1000

  def calc_ventilation_losses(self, iat: float, oat: float,
                              window_open: bool) -> float:
    q = 1.204  #kg/m3 or 1.204/1000 kg/l
    cp = 1005  # J/kg.K
    if window_open:
      mass_flow_rate = q * self.air_change_rate_window_open / 3600 * self.volume_rooms
    else:
      mass_flow_rate = q * self.air_change_rate / 3600 * self.volume_rooms
    return (iat - oat) * mass_flow_rate * cp / 1000  #kW

  def window_opening_schedule(self, OAT: float, IAT: float) -> bool:
    # i. Start to open when the internal temperature exceeds 22°C.
    # ii. Be fully open when the internal temperature exceeds 26°C.
    # iii. Start to close when the internal temperature falls below 26°C.
    # iv. Be fully closed when the internal temperature falls below 22°C.
    window_open = False
    if OAT < IAT and IAT > self.opening_window_iat_limit and OAT < self.opening_window_oat_limit:
      window_open = True
    return window_open

  def run_model(self) -> pd.DataFrame:
    #https://www.sciencedirect.com/science/article/pii/S0360544213005525
    dataf = self.model_data.copy()
    indoor_air_temperatures: npt.NDArray[np.float64] = dataf[
        schema.DataSchema.IAT].to_numpy()
    outdoor_air_temperatures: npt.NDArray[np.float64] = dataf[
        schema.DataSchema.OAT].to_numpy()
    heating_system_outputs: npt.NDArray[np.float64] = dataf[
        schema.DataSchema.HEATINGOUTPUT].to_numpy()
    heat_gains: npt.NDArray[np.float64] = dataf[
        schema.DataSchema.TOTALGAINS].to_numpy()
    ventilation_losses: npt.NDArray[np.float64] = np.zeros(len(dataf))

    for t in np.arange(1, len(dataf.index)):
      temp_iat = indoor_air_temperatures[t - 1]
      temp_oat = outdoor_air_temperatures[t - 1]
      window_open = self.window_opening_schedule(temp_oat, temp_iat)
      ventilation_losses[t] = self.calc_ventilation_losses(
          temp_iat, temp_oat, window_open)
      temp_gains = (heat_gains[t] - ventilation_losses[t])

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
    if heating_season_flag:
      estimated_heating_required = (self.heating_target_indoor_air_temperature
                                    - estimated_iat) / (self.R *
                                                        (1 - self.exp_factor))
    else:
      estimated_heating_required = (self.cooling_target_indoor_air_temperature
                                    - estimated_iat) / (self.R *
                                                        (1 - self.exp_factor))
    estimated_heating_required = self.cap_estimated_heating_system_output(
        estimated_heating_required, heating_season_flag)

    #calculate the expected indoor air temperature
    estimated_iat = estimated_iat + self.R * (
        1 - self.exp_factor) * estimated_heating_required
    if estimated_iat > sim_param.MAX_INDOOR_AIR_TEMP or estimated_iat > self.cooling_target_indoor_air_temperature:
      if not heating_season_flag:  #cooling season
        #Cooling required to avoid going above threshold
        estimated_heating_required = self.max_cooling_output
      else:
        estimated_heating_required = 0

    elif estimated_iat < sim_param.MIN_INDOOR_AIR_TEMP or estimated_iat < self.heating_target_indoor_air_temperature:
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
