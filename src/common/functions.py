from datetime import datetime
from pathlib import Path

import icecream as ic
import numpy as np
import pandas as pd
import xarray as xr

from common import enums, schema, sim_param
from models import thermal_model

PATH_ORG = Path(
    r'D:\Profile data Cardiff\Cardiff University\Energy Data - Documents')

PATH_GEO_LOOKUP = Path(
    r'D:\Profile data Cardiff\OneDrive - Cardiff University\04 - Projects\22 - Heat demand scotland\data\geo_lookup_tables\PCD_OA_LSOA_MSOA_LAD_AUG19_UK_LU\PCD_OA_LSOA_MSOA_LAD_AUG19_UK_LU.csv'
)


def standardise_str(dataf: pd.DataFrame, target_column: str) -> pd.DataFrame:
  new_df = dataf.copy()
  new_df[target_column] = new_df[target_column].str.strip().str.lower(
  ).str.replace('-', '')
  new_df[target_column] = new_df[target_column].fillna("uncategorized")
  return new_df


def get_LSOA_code_to_LADCD_lookup() -> pd.DataFrame:
  geo_lookup = pd.read_csv(PATH_GEO_LOOKUP,
                           encoding='ISO-8859-1',
                           low_memory=False)
  geo_lookup = standardise_str(geo_lookup, schema.geoLookupSchema.ladnm)
  columns_to_keep = [
      schema.geoLookupSchema.ladcd, schema.geoLookupSchema.ladnm,
      schema.geoLookupSchema.lsoa
  ]
  geo_lookup = geo_lookup.loc[:, columns_to_keep].copy()
  return geo_lookup.drop_duplicates().reset_index(drop=True)


def get_LA_code_to_name_lookup() -> pd.DataFrame:
  geo_lookup = pd.read_csv(PATH_GEO_LOOKUP,
                           encoding='ISO-8859-1',
                           low_memory=False)
  geo_lookup = standardise_str(geo_lookup, schema.geoLookupSchema.ladnm)
  columns_to_keep = [
      schema.geoLookupSchema.ladcd, schema.geoLookupSchema.ladnm
  ]
  geo_lookup = geo_lookup.loc[:, columns_to_keep].copy()
  return geo_lookup.drop_duplicates().reset_index(drop=True)


def prepare_residential_data(dataf: pd.DataFrame, init_year: int,
                             target_year: int) -> pd.DataFrame:
  cibse_city_to_region = {
      'Manchester': ['North West'],
      'Birmingham': [
          'West Midlands', 'East', 'South East', 'South West', 'London'
      ],  #South East, West and London are under Birmingham until CIBSE issue is fixed 
      'Cardiff': ['Wales'],
      'Edinburgh': [
          'North East Scotland', 'East Scotland', 'Orkney', 'Borders Scotland',
          'Shetland'
      ],
      'Glasgow': ['West Scotland', 'Highland', 'Western Isles'],
      'Leeds': ['Yorkshire and The Humber', 'North East', 'East Midlands'],
      # 'London_GTW': [],
      # 'London_LHR': ['South West'],
      # 'London_LWC': ['London'],
  }
  lookup_map = {}
  for k, v in cibse_city_to_region.items():
    for region in v:
      lookup_map[region] = k
  pd.DataFrame(
      lookup_map,
      index=[
          schema.DwellingDataSchema.REGION,
          schema.DwellingDataSchema.CIBSE_CITY
      ]).T.to_csv(PATH_ORG /
                  r'General\communication\tables\Region_to_CIBSE_city.csv')
  dataf[schema.DwellingDataSchema.CIBSE_CITY] = dataf[
      schema.DwellingDataSchema.REGION].apply(lambda x: lookup_map[x])

  increase_rate = get_percentage_increase_dwellings(init_year, target_year)
  dataf[schema.DwellingDataSchema.NB_DWELLINGS] = dataf[
      schema.DwellingDataSchema.NB_DWELLINGS].apply(
          lambda x: round(x * increase_rate, 0)).astype(int)

  return dataf


def format_summary_results(results: pd.DataFrame,
                           nb_dwellings: list[int]) -> pd.DataFrame:
  ic.ic(results)
  results.columns = [
      schema.ResultSchema.SPECIFICHEATINGDEMAND_DWELLING,
      schema.ResultSchema.SPECIFICCOOLINGDEMAND_DWELLING
  ]
  results[[
      schema.ResultSchema.HEATINGDEMAND, schema.ResultSchema.COOLINGDEMAND
  ]] = results.mul(nb_dwellings, axis=0)
  results.index.name = schema.ResultSchema.INDEX
  return results


def run_batch_simulation(
    dwellings_data: pd.DataFrame, input_data: pd.DataFrame,
    saving_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
  list_nb_dwellings = dwellings_data[
      schema.DwellingDataSchema.NB_DWELLINGS].values
  metadata_frames = []
  summary_results_frames = []
  LA_str = dwellings_data[schema.DwellingDataSchema.LADNM].unique()[0]
  LACD_str = dwellings_data[schema.DwellingDataSchema.LADCD].unique()[0]
  save_simulation_path = saving_path / f'{LA_str}_{LACD_str}'
  save_simulation_path.mkdir(parents=True, exist_ok=True)
  results_frames = {}
  for ii, row in dwellings_data.iterrows():
    ic.ic(ii)
    results_df = run_simulation(input_data.copy(),
                                row,
                                initial_indoor_air_temperature=21)
    temp_nb_dwellings = row[schema.DwellingDataSchema.NB_DWELLINGS]
    metadata_frames.append(row)
    summary_results_frames.append(print_heating_and_cooling_demand(results_df))
    # results_df.to_parquet(save_simulation_path / f'{ii}_sim_results.gzip')
    results_frames[ii] = results_df[
        schema.DataSchema.HEATINGOUTPUT] * temp_nb_dwellings
  heating_output_df = pd.DataFrame(results_frames)
  heating_output_df.to_csv(save_simulation_path /
                           f'{LA_str}_{LACD_str}_total_heating_outputs.csv')
  # pd.concat(results_frames, axis=1).to_parquet(save_simulation_path /
  #                                              f'{LA_str}_heating_outputs.gzip')
  summary_results = format_summary_results(
      pd.DataFrame(summary_results_frames), list_nb_dwellings)
  summary_results.index = dwellings_data.index
  metadata = pd.concat(metadata_frames, axis=1).T
  metadata.index.name = schema.ResultSchema.INDEX

  return summary_results, metadata


def format_weather_data(dataf: pd.DataFrame) -> pd.DataFrame:
  external_data = dataf.rename(columns={
      'GSR': schema.DataSchema.SOLARRADIATION,
      'DBT': schema.DataSchema.OAT
  })
  external_data['index'] = pd.to_datetime(
      external_data[['Year', 'Month', 'Day', 'Hour']])
  external_data = external_data.set_index('index')
  external_data.head()
  filt = (external_data.index.month >= 5) & (external_data.index.month <= 9)
  cols_to_keep = [schema.DataSchema.SOLARRADIATION, schema.DataSchema.OAT]
  external_data = external_data.loc[filt, cols_to_keep]
  return external_data


def get_percentage_increase_dwellings(init_year: int,
                                      target_year: int) -> float:
  PATH_TABLES = PATH_ORG / r"General\communication\tables"

  fn = "Dwellings_size.csv"
  dataf = pd.read_csv(PATH_TABLES / fn, index_col=0, thousands=r',')
  dataf = dataf.dropna(how='any').T
  str_init_year = str(init_year)
  str_target_year = str(target_year)
  if str_init_year not in dataf.index:
    str_init_year = dataf.index[0]
  if str_target_year not in dataf.index:
    str_target_year = dataf.index[-1]

  init_number = dataf.loc[str_init_year, ' Number of households ']
  target_number = dataf.loc[str_target_year, ' Number of households ']
  return target_number / init_number


def resample_modelling_results(
    simulation_results: pd.DataFrame) -> pd.DataFrame:
  """Resample the results to decrease amount of data and for visualisation purposes."""
  simulation_results[
      schema.DataSchema.
      TIME_SECONDS] = simulation_results.index.values // sim_param.TIMESTEP_SIMULATION
  return simulation_results.groupby(schema.DataSchema.TIME_SECONDS).mean()


def run_simulation(
    sim_data: pd.DataFrame,
    dwelling_data: pd.Series,
    initial_indoor_air_temperature: float | None = 21) -> pd.DataFrame:
  # ) -> list[float]:
  """Estimate the heating and cooling demand for a dwelling
    with specific R and C based on external air temperature and solar radiation."""
  assert initial_indoor_air_temperature is not None
  # Create models

  dwelling = thermal_model.ThermalModel(
      R=1 / dwelling_data[schema.DwellingDataSchema.THERMAL_LOSSES],
      C=dwelling_data[schema.DwellingDataSchema.THERMAL_CAPACITY],
      floor_area=dwelling_data[schema.DwellingDataSchema.FLOOR_AREA],
  )

  dwelling.load_model_data(sim_data)
  rcmodel_dataf = dwelling.run_model()
  rcmodel_dataf.index.name = schema.DataSchema.TIME_HOURS

  return rcmodel_dataf
  # heating_demand, cooling_demand = print_heating_and_cooling_demand(
  #     rcmodel_dataf)
  # return heating_demand, cooling_demand


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
    simulation_results: pd.DataFrame, ) -> tuple[float, float]:
  cooling_demand = float(simulation_results.loc[
      simulation_results[schema.DataSchema.HEATINGOUTPUT] < 0,
      schema.DataSchema.HEATINGOUTPUT,
  ].sum())
  heating_demand = float(simulation_results.loc[
      simulation_results[schema.DataSchema.HEATINGOUTPUT] > 0,
      schema.DataSchema.HEATINGOUTPUT,
  ].sum())

  ic.ic(heating_demand, cooling_demand)
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
  path_netcdf_files = Path(
      path_project) / "data" / f"{target_area}_netcdf_data"
  if path_netcdf_files.exists():
    # Load data
    radiation_data = load_netcdf_files(path_netcdf_files, target_variable)
    radiation_data = radiation_data - radiation_data.shift()
    radiation_data.fillna(0, inplace=True)
    radiation_data[radiation_data < 0] = 0
    radiation_data[target_variable.column_name] = (
        radiation_data[target_variable.column_name] / 3600
    )  # convert from J/m2 to W/m2
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
      )  # convert from degreeK to degreeC
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
              lambda x: calculate_heating_degree(x, HDD_ref_temperature
                                                 )).values / timestep_per_day)
  degree_day_dataf[
      f"{schema.OutputDataSchema.CDD}_{CDD_ref_temperature:0.1f}"] = (
          dataf[schema.OutputDataSchema.OAT].apply(
              lambda x: calculate_cooling_degree(x, CDD_ref_temperature
                                                 )).values / timestep_per_day)
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


def get_lsoa_level_results(dataf: pd.DataFrame,
                           index_lsoa_dict: dict[int, str]) -> pd.DataFrame:
  tuples = [(index_lsoa_dict[int(x)], int(x)) for x in dataf.columns]
  new_columns = pd.MultiIndex.from_tuples(
      tuples, names=[schema.ResultSchema.LSOA, schema.ResultSchema.INDEX])
  dataf.columns = new_columns
  dataf = dataf.T.groupby(level=0).sum().T
  return dataf


def extract_cooling_demand_profiles_and_peaks(path: Path,
                                              residential_data: pd.DataFrame):
  """From the simulation results, aggregate the values at LA level to create an hourly profile of the cooling demand+extract the peak demand for every lsoas in the LA"""
  pathlist = (path / 'simulation').rglob('*_total_heating_outputs.csv')

  index_lsoa_dict: dict[int, str] = residential_data[
      schema.ResultSchema.LSOA].to_dict()

  lsoa_peak_frames: dict[str, pd.Series] = {}
  index_demand_frames: dict[int, pd.Series] = {}
  frames: dict[str, pd.Series] = {}
  for temp_path in pathlist:
    ic.ic(temp_path)
    la_str = temp_path.stem.split('_total_heating_outputs')[0]
    la_code = la_str.split('_')[-1]
    temp_sim_results = pd.read_csv(temp_path, index_col=0, parse_dates=True)
    frames[la_code] = -temp_sim_results.sum(axis=1)
    index_demand_frames[la_code] = -temp_sim_results.sum()
    lsoa_level_results = get_lsoa_level_results(-temp_sim_results,
                                                index_lsoa_dict)
    lsoa_peak_frames[la_code] = lsoa_level_results.max()

  lsoa_peak = pd.concat(lsoa_peak_frames).to_frame()
  lsoa_peak.columns = [schema.VisualisationSchema.PEAK_COOLING]
  index_demand = pd.concat(index_demand_frames).to_frame()
  index_demand.columns = [schema.VisualisationSchema.COOLINGDEMAND]
  cooling_demand = pd.DataFrame(frames)
  cooling_demand.to_csv(path / 'cooling_demand_profiles_la.csv')
  lsoa_peak.to_csv(path / 'cooling_peak_results_lsoa.csv')
  return lsoa_peak, cooling_demand
