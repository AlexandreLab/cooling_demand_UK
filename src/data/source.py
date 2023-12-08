from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import schema, sim_param

SIM_DATA_COLUMNS = {
    schema.DataSchema.IAT: 'float64',
    schema.DataSchema.OAT: 'float64',
    schema.DataSchema.HEATINGOUTPUT: 'float64',
    schema.DataSchema.SOLARRADIATION: 'float64',
    schema.DataSchema.SOLARGAINS: 'float64',
    schema.DataSchema.OCCUPANCYGAINS: 'float64',
    schema.DataSchema.APPLIANCESGAINS: 'float64',
    schema.DataSchema.TOTALGAINS: 'float64',
    schema.DataSchema.IHG: 'float64',
    schema.DataSchema.HEATINGSEASON: int,
}


@dataclass
class SimulationData:
  weather_data: pd.DataFrame
  sim_data: pd.DataFrame = field(init=False)
  CDD_ref_temperature: float = 24.
  HHD_ref_temperatre: float = 15.5

  def __post_init__(self):
    cooling_weeks = self.get_cooling_weeks(self.CDD_ref_temperature)
    self.add_heating_season_flag(cooling_weeks)

  @property
  def all_months(self) -> list[int]:
    return self.weather_data.index.month.unique().to_list()

  @property
  def all_years(self) -> list[int]:
    return self.weather_data.index.year.unique().to_list()

  @property
  def all_weeks(self) -> list[int]:
    return list(self.weather_data.index.isocalendar().week.unique())

  def filter_weather_data(self,
                          list_weeks: list[int] | None = None,
                          list_months: list[int] | None = None,
                          list_years: list[int] | None = None) -> pd.DataFrame:
    """Return a filtered copy of the data"""
    if list_weeks is None:
      list_weeks = self.all_weeks
    if list_months is None:
      list_months = self.all_months
    if list_years is None:
      list_years = self.all_years

    filt = (self.weather_data.index.year.isin(list_years)
            & self.weather_data.index.month.isin(list_months)
            & self.weather_data.index.isocalendar().week.isin(list_weeks))
    return self.weather_data.loc[filt, :].copy()

  def get_cooling_weeks(self, CDD_ref_temperature: float) -> list[int]:
    """Return a list of the weeks for which cooling is required. Default is all weeks if no CDD column."""
    cdd_col = f"{schema.OutputDataSchema.CDD}_{CDD_ref_temperature}"
    if cdd_col in self.weather_data.columns:
      cooling_seasons: list[int] = list(self.weather_data.loc[
          self.weather_data[cdd_col] > 0].index.isocalendar().week.unique())
    else:
      cooling_seasons: list[int] = list(
          self.weather_data.index.isocalendar().week.unique())
    cooling_seasons = list(
        np.arange(min(cooling_seasons),
                  max(cooling_seasons) + 1))
    return cooling_seasons

  def add_heating_season_flag(self,
                              weeks_with_cooling: list[int] | None = None
                              ) -> None:
    """Add heating/cooling season flag within the dataset"""
    self.weather_data[schema.OutputDataSchema.HEATINGSEASON] = 1
    self.weather_data.loc[
        self.weather_data.index.isocalendar().week.isin(weeks_with_cooling),
        schema.OutputDataSchema.HEATINGSEASON] = 0

  def create_simulation_data_skeleton(
      self, org_date_index: pd.DatetimeIndex) -> pd.DataFrame:
    dataf = pd.DataFrame(
        {
            col_name: pd.Series(dtype=col_type)
            for col_name, col_type in SIM_DATA_COLUMNS.items()
        },
        index=org_date_index)
    # dataf.resample(sim_param.TIMESTEP_SIMULATION).mean()

    return dataf

  def create_default_simulation_data(self,
                                     dataf: pd.DataFrame,
                                     initial_IAT: float = 21) -> pd.DataFrame:
    """Create default data considering fixed indoor and outdoor air temperature, no heating output and no solar gains."""
    for col in SIM_DATA_COLUMNS.keys():
      dataf[col] = 0
    initial_index = dataf.index[0]
    dataf.loc[initial_index, schema.DataSchema.IAT] = initial_IAT
    dataf[schema.DataSchema.OAT] = 24
    dataf = dataf.astype(float)
    return dataf

  # def calculate_ventilation_losses(self, wind_speed:float) -> float:
  #   """"Ventilation losses in kW (SAP 2012, p113)"""
  #   infiltration_rate = wind_speed/4
  #   volume = # Volume of rooms
  #   n = # air change rate
  #   return 0.33*volume*n


# def estimate_cooling_demand(R:float, degree_days:float, COP:float)->float:
#     """Estimate the number of energy to heat/cool a dwelling based on the number of degree days.
#     -R in [kW/K]
#     -degree_days [K]
#     -COP: efficient of heating/cooling system"""
#     return R*degree_days*24/COP

  def create_CIBSE_based_simulation_data(
      self,
      initial_IAT: float = 21,
      list_weeks: list[int] | None = None,
      list_months: list[int] | None = None,
      list_years: list[int] | None = None) -> pd.DataFrame:

    filtered_weather_data = self.filter_weather_data(list_weeks=list_weeks,
                                                     list_months=list_months,
                                                     list_years=list_years)
    dataf = (self.create_simulation_data_skeleton(
        filtered_weather_data.index).pipe(self.create_default_simulation_data,
                                          initial_IAT))
    dataf.loc[:, schema.DataSchema.OAT] = filtered_weather_data[
        schema.DataSchema.OAT].values

    dataf.loc[:, schema.DataSchema.SOLARRADIATION] = np.nan
    dataf[schema.DataSchema.SOLARRADIATION] = filtered_weather_data[
        schema.DataSchema.SOLARRADIATION].values
    dataf[schema.DataSchema.SOLARRADIATION].fillna(
        dataf[schema.DataSchema.SOLARRADIATION].interpolate(), inplace=True)
    return dataf

  def create_era5_based_simulation_data(
      self,
      initial_IAT: float = 21,
      list_weeks: list[int] | None = None,
      list_months: list[int] | None = None,
      list_years: list[int] | None = None) -> pd.DataFrame:

    filtered_era5_data = self.filter_weather_data(list_weeks=list_weeks,
                                                  list_months=list_months,
                                                  list_years=list_years)

    dataf = (self.create_simulation_data_skeleton(
        filtered_era5_data.index).pipe(self.create_default_simulation_data,
                                       initial_IAT))
    filtered_era5_data = filtered_era5_data.resample(
        sim_param.TIMESTEP_SIMULATION).mean()

    dataf[schema.DataSchema.OAT] = filtered_era5_data[
        schema.DataSchema.OAT].values
    dataf[schema.DataSchema.HEATINGSEASON] = filtered_era5_data[
        schema.DataSchema.HEATINGSEASON].values
    dataf[schema.DataSchema.HEATINGSEASON].fillna(method='ffill', inplace=True)

    dataf.loc[:, schema.DataSchema.SOLARRADIATION] = np.nan
    dataf[schema.DataSchema.SOLARRADIATION] = filtered_era5_data[
        schema.DataSchema.SOLARRADIATION].values
    dataf[schema.DataSchema.SOLARRADIATION].fillna(
        dataf[schema.DataSchema.SOLARRADIATION].interpolate(), inplace=True)

    return dataf
