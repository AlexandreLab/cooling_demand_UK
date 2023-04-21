from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..data import source
from ..models import thermal_model
from . import enums, schema

TIMESTEP_SIMULATION = 3600


def run_simulation(
    external_data: pd.DataFrame,
    dwelling_data: pd.Series,
    simulation_year: int,
    solar_gains: bool,
    initial_indoor_air_temperature: float | None = 21,
) -> list[float]:
  """Estimate the heating and cooling demand for a dwelling
    with specific R and C based on external air temperature and solar radiation."""
  assert initial_indoor_air_temperature is not None
  # Create models

  dwelling = thermal_model.ThermalModel(
      R=1 / dwelling_data["Average thermal losses kW/K"],
      C=dwelling_data["Average thermal capacity kJ/K"],
      floor_area=dwelling_data["Average floor area m2"],
      initial_indoor_air_temperature=initial_indoor_air_temperature,
  )
  dwelling.init_parameters()
  data_source = source.SimulationData(dwelling,
                                      external_data,
                                      timestep_simulation=TIMESTEP_SIMULATION)
  rcmodel_dataf = data_source.create_era5_based_simulation_data(
      estimate_solar_gains=solar_gains, list_years=[simulation_year])
  rcmodel_dataf = dwelling.estimate_heating_demand(rcmodel_dataf)
  resampled_rcmodel_dataf = data_source.resample_modelling_results(
      rcmodel_dataf)
  heating_demand, cooling_demand = print_heating_and_cooling_demand(
      resampled_rcmodel_dataf)
  return [heating_demand, cooling_demand]


def create_dd_dataframes_for_all_LAs(
    list_LAs: list[str],
    path_results: Path,
    HDD_ref_temperature: float,
    CDD_ref_temperature: float,
) -> pd.DataFrame:
  """Create dataframes including data from the netcdf files related
    to the LA and calculate the HDD and CDD."""
  degree_day_dataf = pd.DataFrame()
  for LA_str in list_LAs:
    print(f"Extracting data for {LA_str}")
    input_data = get_all_data(LA_str)
    degree_day_dataf = get_degree_days_dataframe(input_data,
                                                 HDD_ref_temperature,
                                                 CDD_ref_temperature)
    # export results
    degree_day_dataf.to_csv(
        Path(path_results) / "raw" /
        f"{LA_str}_degree_days.csv".replace(" ", "_"))
  return degree_day_dataf


def print_heating_and_cooling_demand(
    simulation_results: pd.DataFrame,) -> tuple[float, float]:
  cooling_demand = float(simulation_results.loc[
      simulation_results[schema.DataSchema.HEATINGOUTPUT] < 0,
      schema.DataSchema.HEATINGOUTPUT,].sum())
  heating_demand = float(simulation_results.loc[
      simulation_results[schema.DataSchema.HEATINGOUTPUT] > 0,
      schema.DataSchema.HEATINGOUTPUT,].sum())

  print(
      f"heating demand {heating_demand}kWh and cooling demand is {cooling_demand}kWh"
  )
  return heating_demand, cooling_demand


def apply_outputdataschema(dataf: pd.DataFrame,
                           target_variable: enums.ExtractFile) -> pd.DataFrame:
  if target_variable is enums.ExtractFile.TEMPERATURE:
    dataf.index.name = schema.OutputDataSchema.DATEINDEX
    dataf.columns = [schema.OutputDataSchema.OAT]
  elif target_variable is enums.ExtractFile.SOLARRADIATION:
    dataf.index.name = schema.OutputDataSchema.DATEINDEX
    dataf.columns = [schema.OutputDataSchema.SOLARRADIATION]
  else:
    print(f"Schema not recognized for target variable: {target_variable}")
  return dataf


def get_all_data(target_area: str) -> pd.DataFrame:
  frames = []
  functions = [get_radiation_data, get_temperature_data]
  for f in functions:
    frames.append(f(target_area))

  return pd.concat(frames, axis=1)


def get_radiation_data(target_area: str) -> pd.DataFrame:

  target_variable = enums.ExtractFile.SOLARRADIATION

  path_project: str = str(Path().absolute()).split("code")[0]
  path_netcdf_files = Path(path_project) / "data" / f"{target_area}_netcdf_data"
  if path_netcdf_files.exists():
    # Load data
    radiation_data = load_netcdf_files(path_netcdf_files, target_variable)
    radiation_data = radiation_data - radiation_data.shift()
    radiation_data.fillna(0, inplace=True)
    radiation_data[radiation_data < 0] = 0
    radiation_data[target_variable.column_name] = (
        radiation_data[target_variable.column_name] / 3600
    )    # convert from J/m2 to W/m2
  else:
    radiation_data = pd.DataFrame(columns=[target_variable.column_name])
    print(f"The radiation data for {target_area} could not be found.")

  return apply_outputdataschema(radiation_data, target_variable)


def get_temperature_data(target_area: str) -> pd.DataFrame:
  temperature_data = pd.DataFrame()
  target_variable = enums.ExtractFile.TEMPERATURE

  if target_area == "NY":
    # Load NY PUMAs data
    path_data: Path = Path().absolute().parent / "data" / "input_data"
    filename: str = "temps_pumas_NY_2019.csv"
    full_path = path_data / filename
    temperature_data = load_data(full_path).pipe(add_datetime_index)
  else:
    path_project: str = str(Path().absolute()).split("code")[0]
    path_netcdf_files = Path(
        path_project) / "data" / f"{target_area}_netcdf_data"
    if path_netcdf_files.exists():
      temperature_data = load_netcdf_files(path_netcdf_files, target_variable)
      temperature_data[target_variable.column_name] = (
          temperature_data[target_variable.column_name] - 273.15
      )    # convert from degreeK to degreeC
    else:
      print(f"The temperature data for {target_area} could not be found.")

  return apply_outputdataschema(temperature_data, target_variable)


def load_data(path: Path) -> pd.DataFrame:
  dataf = pd.read_csv(str(path))
  return dataf


def add_datetime_index(dataf: pd.DataFrame, reference_year: int = 2021):
  start_date = datetime(reference_year, 1, 1)
  end_date = datetime(reference_year + 1, 1, 1)
  index = pd.date_range(start=start_date, end=end_date, freq="1h")[:-1]
  dataf.index = index

  return dataf


def calculate_MAE(org_serie: pd.Series, model_serie: pd.Series) -> float:
  return np.sum(np.abs(org_serie - model_serie)) / len(org_serie)


def add_time_features(dataf):
  dataf['Hour'] = dataf.index.hour
  dataf['Day_of_week'] = dataf.index.dayofweek
  dataf['Day'] = dataf.index.dayofyear
  dataf['Month'] = dataf.index.month
  dataf['Year'] = dataf.index.year
  dataf['Weekday_flag'] = [1 if x < 5 else 0 for x in dataf.index.dayofweek]
  dataf['HH'] = dataf.index.hour * 2 + dataf.index.minute // 30
  dataf['HH'] = dataf['HH'].astype(int)
  dataf['Date'] = dataf.index.date
  dataf['Week'] = dataf.index.isocalendar().week
  return dataf


def calculate_heating_degree(current_temperature: float,
                             ref_temperature: float) -> float:
  hdd = ref_temperature - current_temperature
  if hdd < 0:
    hdd = 0
  return hdd


def calculate_cooling_degree(current_temperature: float,
                             ref_temperature: float) -> float:
  cdd = current_temperature - ref_temperature
  if cdd < 0:
    cdd = 0
  return cdd


def get_degree_days_dataframe(
    dataf: pd.DataFrame,
    HDD_ref_temperature: float,
    CDD_ref_temperature: float,
    timestep_per_day: int = 24,
) -> pd.DataFrame:
  """Add number of degree days based on the temperature in each column"""
  # timestep_per_day = number of steps per day 24 - if hourly resolution, 48 if half-hourly
  degree_day_dataf = dataf.copy()
  degree_day_dataf[
      f"{schema.OutputDataSchema.HDD}_{HDD_ref_temperature:0.1f}"] = (
          dataf[schema.OutputDataSchema.OAT].apply(
              lambda x: calculate_heating_degree(x, HDD_ref_temperature)).values
          / timestep_per_day)
  degree_day_dataf[
      f"{schema.OutputDataSchema.CDD}_{CDD_ref_temperature:0.1f}"] = (
          dataf[schema.OutputDataSchema.OAT].apply(
              lambda x: calculate_cooling_degree(x, CDD_ref_temperature)).values
          / timestep_per_day)
  return degree_day_dataf


def get_rolling_average_daily_degree_days_dataf(
    dataf: pd.DataFrame, target_col: str, number_of_days: int) -> pd.DataFrame:
  """calculate the average daily heating/cooling degree days using a rolling window"""
  return (dataf[target_col].rolling(
      window=24 * number_of_days, min_periods=1, center=True).sum() /
          number_of_days)


def load_netcdf_files(path_directory: Path,
                      target_variable: enums.ExtractFile) -> pd.DataFrame:
  """Load netcdf file and transform them into dataframes."""
  pathlist = path_directory.rglob("*.nc")
  frames: list[pd.DataFrame] = []
  for path in pathlist:
    if target_variable.filename_key in path.stem:
      # https://docs.xarray.dev/en/stable/examples/ERA5-GRIB-example.html
      temp_xarray = xr.open_dataset(path)
      temp_dataf = temp_xarray.to_dataframe()
      temp_dataf = temp_dataf.unstack([0, 1]).iloc[:, 0]
      temp_dataf = temp_dataf.to_frame()
      temp_dataf.columns = temp_dataf.columns.get_level_values(0)
      frames.append(temp_dataf)
  if len(frames) > 0:
    concat_dataf = pd.concat(frames)
    concat_dataf.index = concat_dataf.index.tz_localize("utc")
  else:
    concat_dataf = pd.DataFrame(columns=[target_variable.column_name])
  return concat_dataf
