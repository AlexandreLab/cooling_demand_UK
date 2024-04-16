from pathlib import Path

import icecream as ic
import pandas as pd

from common import functions, schema
from config import settings
from data import source

PATH_ORG = Path(
    r'D:\Profile data Cardiff\Cardiff University\Energy Data - Documents')

PATH_GB_DATA = Path(
    r'D:\Profile data Cardiff\OneDrive - Cardiff University\04 - Projects\00 - Final data\Annual_demand'
)

PATH_CIBSE_DATA = PATH_ORG / r'General\data\CIBSE weather data\WD16SET\WD16SET\WDD16SET\WDD16SET\WMD16SET\WMD16SET'
PATH_RESULTS = PATH_ORG / r'General\04 - Analysis\2050 high emission medium thermal capacity'
PATH_SIMULATION_RESULTS = PATH_RESULTS / r'simulation'
PATH_METADATA = PATH_RESULTS / r'metadata'
PATH_SUMMARY_RESULTS = PATH_RESULTS / r'summary_results'


def import_thermal_characteristics_data(path_data: Path, init_year: int,
                                        target_year: int) -> pd.DataFrame:
  """Import the thermal characteristics data of the building stock."""
  temp_dataf = functions.get_LSOA_code_to_LADCD_lookup()
  temp_dataf = temp_dataf.set_index(schema.geoLookupSchema.lsoa)
  residential_data = pd.read_csv(path_data, index_col=0)
  residential_data = functions.prepare_residential_data(
      residential_data, init_year, target_year)
  residential_data = pd.merge(residential_data,
                              temp_dataf,
                              left_on=schema.DwellingDataSchema.LSOA,
                              right_index=True)
  return residential_data


def estimate_heating_cooling_demand_all_las(init_year: int,
                                            target_year: int) -> None:
  """Estimate the heating/cooling demand for all LAs in England and"""
  path_thermal_data = PATH_GB_DATA / 'Thermal_characteristics_afterEE.csv'
  residential_data = import_thermal_characteristics_data(
      path_thermal_data, init_year, target_year)

  column_names = [
      'Year', 'Month', 'Day', 'Hour', 'PWC', 'Cloud', 'DBT', 'WBT', 'RH',
      'Press', 'WD', 'WS', 'GSR', 'DSR', 'Alt', 'Dec', 'Cloud1', 'DBT1',
      'WBT1', 'Press1', 'WD1', 'WS1'
  ]
  cibse_city_list = [
      'Glasgow', 'Birmingham', 'Cardiff', 'Edinburgh', 'Leeds', 'Manchester'
  ]
  pathlist = Path(PATH_CIBSE_DATA).rglob('*_DSY2_2050High50*.csv')
  for path in pathlist:
    # ic.ic(path.stem)
    cibse_city = path.stem.split('_DSY2_')[0]

    if cibse_city in cibse_city_list:
      ic.ic(cibse_city)
      # list_files.append(path.stem)
      temp_dataf = pd.read_csv(path,
                               skiprows=32,
                               header=None,
                               delimiter=",",
                               names=column_names)
      weather_data = functions.format_weather_data(temp_dataf)
      data_source = source.SimulationData(weather_data)
      sim_dataf = data_source.create_CIBSE_based_simulation_data()
      filt_las = (
          residential_data[schema.DwellingDataSchema.CIBSE_CITY] == cibse_city)
      list_las = list(
          residential_data.loc[filt_las,
                               schema.DwellingDataSchema.LADNM].unique())
      ic.ic(list_las)
      for LA_str in list_las:
        ic.ic(LA_str)
        filt = (
            (residential_data[schema.DwellingDataSchema.LADNM]
             == LA_str.lower())
            &
            (residential_data[schema.DwellingDataSchema.THERMAL_CAPACITY_LEVEL]
             == settings.thermal_capacity_level))
        LA_residential_data = residential_data.loc[filt, :].copy()
        # ic.ic(weather_data)
        ic.ic(LA_residential_data)
        summary_results, metadata = functions.run_batch_simulation(
            LA_residential_data, sim_dataf, PATH_SIMULATION_RESULTS)

        PATH_METADATA.mkdir(parents=True, exist_ok=True)
        metadata.to_parquet(PATH_METADATA / f'{LA_str}_metadata.gzip')
      ic.ic(LA_residential_data)
  return None


def load_csv_file(temp_path: Path) -> pd.DataFrame:
  return pd.read_csv(temp_path, index_col=0, parse_dates=True)


def calculate_results_la_level() -> pd.DataFrame:
  """Calculate the cooling demand at local authority level"""
  pathlist = Path(PATH_SIMULATION_RESULTS).rglob('*.csv')
  frames = {}
  for temp_path in pathlist:
    ic.ic(temp_path)
    temp_sim_results = load_csv_file(temp_path)
    str_la = temp_path.stem.split('_total_heating_outputs')[0]
    frames[str_la] = temp_sim_results.sum(axis=1)
  results = pd.DataFrame(frames)
  PATH_SUMMARY_RESULTS.mkdir(parents=True, exist_ok=True)
  results.to_parquet(PATH_SUMMARY_RESULTS /
                     r'summary_local_autority_results.csv')
  return results


def main():
  """Main function"""
  ic.ic("Main")
  init_year = 2020
  target_year = 2050
  estimate_heating_cooling_demand_all_las(init_year, target_year)
  calculate_results_la_level()


if __name__ == "__main__":
  main()
