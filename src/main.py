from pathlib import Path

import pandas as pd

from common import functions, schema


def get_all_LAs(dataf: pd.DataFrame) -> list[str]:
  """Return a list of the local authorities in the given dataframe."""
  return dataf["Local Authority"].unique().tolist()


def filter_thermal_characteristics_data(
    dataf: pd.DataFrame,
    las_list: list[str] | None = None,
    thermal_capacity_level: str | None = "medium",
):
  """Import the thermal characteristics data for a given LA."""
  if las_list is None:
    las_list = get_all_LAs(dataf)
  filt = (dataf["Local Authority"].isin(las_list)) & (
      dataf["Thermal capacity level"] == thermal_capacity_level)
  return dataf.loc[filt, :].copy()


def import_thermal_characteristics_data(path_data: Path):
  """Import the thermal characteristics data for a given LA."""
  residential_data = pd.read_csv(path_data, index_col=0)
  return residential_data


def import_external_data(path_data: Path):
  """Import the temperature and solar radiation data for a given LA."""
  external_data = pd.read_csv(path_data, index_col=0, parse_dates=True)
  external_data.fillna(0, inplace=True)
  return external_data


def estimate_heating_cooling_demand_all_las() -> None:
  """Estimate the heating/cooling demand for all LAs in England and"""
  # Parameters
  target_year: int = 2022
  solar_gains = False
  path_thermal_data = Path(
      r"../data/input_data/Thermal_characteristics_beforeEE.csv")
  path_externa_data = Path().absolute().parent / "data"
  path_results_simulation = Path().absolute().parent / "data" / "results"
  path_results_simulation.mkdir(parents=True, exist_ok=True)

  residential_data = import_thermal_characteristics_data(path_thermal_data)

  for LA_str in get_all_LAs(residential_data)[:2]:
    print(LA_str)
    temp_path_external_data = (path_externa_data / "raw" /
                               f"{LA_str}_degree_days.csv".replace(" ", "_"))
    external_data = import_external_data(temp_path_external_data)
    filtered_residential_data = filter_thermal_characteristics_data(
        residential_data, [LA_str])

    temp_LA_results = filtered_residential_data.apply(
        lambda row: functions.run_simulation(
            external_data,
            1 / row["Average thermal losses kW/K"],
            row["Average thermal capacity kJ/K"],
            target_year,
            solar_gains,
        ),
        axis=1,
        result_type="expand",
    )
    temp_LA_results.columns = [
        schema.ResultSchema.HEATINGDEMAND,
        schema.ResultSchema.COOLINGDEMAND,
    ]
    temp_LA_results[schema.ResultSchema.YEAR] = target_year
    temp_LA_results.index.name = "Index"
    temp_LA_results.to_csv(
        path_results_simulation /
        f"{LA_str}_heating_cooling_demand.csv".replace(" ", "_"))
    return None


def main():
  """Main function"""
  print("Main")
  estimate_heating_cooling_demand_all_las()


if __name__ == "__main__":
  main()
