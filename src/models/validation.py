from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.common import enums, functions, schema

PATH_MEASURED_IAT = Path(
    r'C:\Users\sceac10\OneDrive - Cardiff University\General\resources\prediction of overheating data\MeasuredTemperature_16Jun-6Jul_2017.xlsx'
)
PATH_EXTENDED_MEASURED_IAT = Path(
    r"C:\Users\sceac10\OneDrive - Cardiff University\General\resources\prediction of overheating data\extra_data_Loughborough"
)


def calculate_cooling_degree_hours(IAT_df: pd.Series,
                                   threshold: float = 24.) -> float:
  """Calculate the number of cooling degree hours of a dwelling based on its inside air temperature"""
  arr_values = IAT_df.values
  arr_values = arr_values - threshold
  arr_values = np.where(arr_values > 0, arr_values, 0)
  return np.sum(arr_values)


def get_extended_measured_data(filename: str) -> pd.DataFrame:
  measured_data = pd.read_csv(PATH_EXTENDED_MEASURED_IAT / filename,
                              index_col=0,
                              parse_dates=True)
  if 'West' in filename:
    cols_to_keep = [
        'U01_living', 'U09_kitchen', 'U13_frontbedroom_centre_1.1m',
        'U27_rearbedroom', 'U33_singlebedroom', 'U35_bathroom', 'U37_landing',
        'U11_hall', 'U07_dining'
    ]
  else:
    cols_to_keep = [
        'U41_living', 'U49_kitchen', 'U47_dining', 'U51_hall',
        'U53_frontbedroom_centre_1.1m', 'U70_rearbedroom', 'U79_singlebedroom',
        'U81_bathroom', 'U83_landing'
    ]

  return measured_data[cols_to_keep].copy()


def get_measured_data_excel(list_name_sheets: list[str]) -> pd.DataFrame:
  frames = {}
  if len(list_name_sheets) > 0:
    for sheet_name in list_name_sheets:
      temp_dataf = pd.read_excel(PATH_MEASURED_IAT,
                                 sheet_name=sheet_name,
                                 header=1,
                                 index_col=1)
      temp_dataf.rename(columns={'Measured': schema.DataSchema.IAT},
                        inplace=True)
      temp_dataf[schema.DataSchema.IAT] = temp_dataf[schema.DataSchema.IAT]
      frames[sheet_name] = temp_dataf[schema.DataSchema.IAT].values
  return pd.DataFrame(frames)


@dataclass
class validation_RC_model:
  """Class used to compare and visualise the results of the simulation and compare it with the measured data from the
  paper: https://journals.sagepub.com/doi/full/10.1177/0143624419847349#sec-10
  """
  simulation_data: pd.DataFrame
  param_dwellings: dict

  @property
  def filename(self) -> str:
    return self.param_dwellings['filename']

  @property
  def extended_dataset(self) -> bool:
    return self.param_dwellings['extended_dataset']

  @property
  def list_sheets(self):
    return self.param_dwellings['sheet_names']

  @property
  def volume_rooms(self):
    return self.param_dwellings['volume_rooms']

  def weighted_average_measured_IAT_dwelling(
      self, measured_data: pd.DataFrame) -> pd.DataFrame:
    total_volume = sum(self.volume_rooms.values())
    weights = [
        self.volume_rooms[col_name] / total_volume
        for col_name in measured_data.columns
    ]
    measured_IAT = measured_data.copy()
    measured_IAT = measured_IAT * weights
    measured_IAT = measured_IAT.sum(axis=1).to_frame()
    measured_IAT.columns = [schema.DataSchema.IAT]
    return measured_IAT

  def calculate_metrics(self) -> tuple[float, float]:
    """Calculate the cooling degree hours and the MAE of the simulation data compared to the measured data"""
    individual_rooms_IAT = self.get_measured_data()
    average_IAT = self.weighted_average_measured_IAT_dwelling(
        individual_rooms_IAT)

    cdh_error = self.get_cooling_degree_hours_error(average_IAT)
    mae = self.get_MAE(average_IAT)
    return cdh_error, mae

  def get_cooling_degree_hours_error(self, average_IAT: pd.DataFrame) -> float:
    modelled_cooling_degree_hours = calculate_cooling_degree_hours(
        self.simulation_data[schema.DataSchema.IAT], 24)
    measured_cooling_degree_hours = calculate_cooling_degree_hours(
        average_IAT[schema.DataSchema.IAT], 24)
    error_cooling_degree_hours = (
        modelled_cooling_degree_hours -
        measured_cooling_degree_hours) / measured_cooling_degree_hours
    print(
        f'number of modelled cooling degree hours {modelled_cooling_degree_hours}'
    )
    print(
        f'number of measured cooling degree hours {measured_cooling_degree_hours}'
    )
    print(
        f"The error in cooling degree hours of the model compared to measured data is {error_cooling_degree_hours:.2%}."
    )
    return error_cooling_degree_hours

  def get_MAE(self, average_IAT: pd.DataFrame) -> float:
    MAE_results = functions.calculate_MAE(
        average_IAT[schema.DataSchema.IAT],
        self.simulation_data[schema.DataSchema.IAT])
    print(
        f"The MAE of the model compared to measured data is {MAE_results:.2f} degreeC."
    )
    return MAE_results

  def get_measured_data(self) -> pd.DataFrame:
    if self.extended_dataset:
      individual_rooms_IAT = get_extended_measured_data(self.filename)
    else:
      individual_rooms_IAT = get_measured_data_excel(self.list_sheets)
    return individual_rooms_IAT

  def plot_measured_data(self, fig, ax, plot_all_rooms: bool = False):
    """Plot the measured data"""
    individual_rooms_IAT = self.get_measured_data()
    average_IAT = self.weighted_average_measured_IAT_dwelling(
        individual_rooms_IAT)
    individual_rooms_IAT.index = self.simulation_data.index
    average_IAT.index = self.simulation_data.index
    average_IAT[schema.DataSchema.IAT].plot(
        ax=ax,
        color='black',
        linewidth=0.8,
        label=enums.DataSource.MEASURED.value)
    if plot_all_rooms:
      for ii, room in enumerate(individual_rooms_IAT.columns):
        individual_rooms_IAT[room].plot(ax=ax,
                                        color=sns.color_palette()[ii],
                                        linewidth=0.5)

    ax.set_ylabel(schema.VisualisationSchema.IAT)
    ax.set_ylim(0, 40)
    ax.margins(0, None)
    ax.legend()
    return fig, ax

  def plot_results(self, fig, ax, plot_all_rooms: bool = False):
    """Plot the results of the simulation and compare it to the measured data."""
    # fig, ax = plt.subplots(figsize=figsize)
    fig, ax = self.plot_measured_data(fig, ax, plot_all_rooms)
    self.simulation_data[schema.DataSchema.IAT].plot(
        ax=ax,
        kind='line',
        color='red',
        marker='*',
        markevery=20,
        linewidth=1,
        label=enums.DataSource.RC.value)
    # ax.legend()
    return fig, ax
