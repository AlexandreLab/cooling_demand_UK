from typing import Callable, List, Tuple

import numpy as np
from scipy.stats import truncnorm

TemperatureFunction = Callable[[int], float]


def random_truncated_normal(mean: float, std: float, lower_bound: float,
                            upper_bound: float) -> float:
  """Return a random value of the truncated normal distribution described."""
  val = np.random.normal(loc=mean, scale=std)
  if val > upper_bound or val < lower_bound:
    val = random_truncated_normal(mean, std, lower_bound, upper_bound)
  return val


def cfd_truncated_normal(x: float, mean: float, std: float, lower_bound: float,
                         upper_bound: float) -> float:
  """Return the cumulative density function at x of the truncated normal distribution described."""
  a, b = (lower_bound - mean) / std, (upper_bound - mean) / std
  return float(truncnorm.cdf(x, a, b, loc=mean, scale=std))


def truncated_normal(x: float, mean: float, std: float, lower_bound: float,
                     upper_bound: float) -> float:
  """Return a function of a truncated normal distribution"""
  if x > upper_bound or x < lower_bound:
    prob_density = 0
  else:
    prob_density = (1 / (std * np.sqrt(2 * np.pi)) *
                    np.exp(-0.5 * ((x - mean) / std)**2))
  return prob_density


def determine_flexibility_service(
    capacity_factor: float,
    lower_temperature_threshold: float,
    upper_temperature_threshold: float,
) -> Tuple[float, bool]:
  """Determine if it is demand reduction or demand increase and the temperature threshold for the modelling"""
  demand_increase = False
  limit_temperature = lower_temperature_threshold
  # default assumptions
  if capacity_factor == 1:
    demand_increase = True
    limit_temperature = upper_temperature_threshold
  elif capacity_factor == 0:
    demand_increase = False
    limit_temperature = lower_temperature_threshold
  else:
    print(
        f"Capacity factor of {capacity_factor} is not recognised. It is assumed that the demand increase value will be {demand_increase}."
    )
  return limit_temperature, demand_increase


def get_heating_systems_efficiency(outside_air_temperature: float) -> float:
  """ Return the average heating systems efficiency based on outside air temperature"""
  efficiency_map = {-5.0: 2, 0.0: 2.3, 5.0: 2.4, 10.0: 2.6}
  return efficiency_map[outside_air_temperature]


def get_mean_inside_air_temperature_func(
    capacity_factor: float,
    lower_temperature_threshold: float = 18,
    upper_temperature_threshold: float = 24,
    mean_indoor_temperature: float = 19.0,
) -> Tuple[TemperatureFunction, float, bool]:

  func = lambda x: mean_indoor_temperature

  limit_temperature, demand_increase = determine_flexibility_service(
      capacity_factor, lower_temperature_threshold,
      upper_temperature_threshold)
  return func, limit_temperature, demand_increase


def get_pdf_inside_air_temperature_func(
    capacity_factor: float,
    lower_temperature_threshold: float = 18,
    upper_temperature_threshold: float = 24,
) -> Tuple[TemperatureFunction, float, bool]:
  """Return a function which is used to generate random inside air temperature of dwellings based on a normal truncated distribution"""

  # default temperature for a stock of dwellings during the heating season.
  # Around 33 to 40% of the dwellings are below the threshold of 18C.
  mean_indoor_temperature: float = 19.0
  std_indoor_temperature: float = 2.5
  max_observed_indoor_temperature: float = 24.0
  min_observed_indoor_temperature: float = 14.0
  # print(
  #     cfd_truncated_normal(
  #         lower_temperature_threshold,
  #         mean_indoor_temperature,
  #         std_indoor_temperature,
  #         min_observed_indoor_temperature,
  #         max_observed_indoor_temperature,
  #     )
  # )
  # print(
  #     1
  #     - cfd_truncated_normal(
  #         upper_temperature_threshold,
  #         mean_indoor_temperature,
  #         std_indoor_temperature,
  #         min_observed_indoor_temperature,
  #         max_observed_indoor_temperature,
  #     )
  # )
  func = lambda x: random_truncated_normal(
      mean_indoor_temperature,
      std_indoor_temperature,
      min_observed_indoor_temperature,
      max_observed_indoor_temperature,
  )

  limit_temperature, demand_increase = determine_flexibility_service(
      capacity_factor, lower_temperature_threshold,
      upper_temperature_threshold)
  return func, limit_temperature, demand_increase


def calculate_max_temperature(outdoor_air_temperature: float,
                              current_heating_output: float,
                              R: float) -> float:
  # maximum inside temperature of the dwelling
  max_temp = current_heating_output * R + outdoor_air_temperature
  #     print(f'The max temperature that can be reached by the system is {max_temp}ºC')
  return round(max_temp, 2)


def calculate_duration_service(
    initial_indoor_air_temperature: float,
    outdoor_air_temperature: float,
    limit_indoor_air_temperature: float,
    current_heating_output: float,
    max_heating_output: float,
    R: float,
    C: float,
) -> float:

  if current_heating_output > max_heating_output:
    duration = np.nan
  elif current_heating_output < 0:
    duration = np.nan
  else:
    # duration codes:
    # -1 = the service can be provided an infinite amount of time
    # else the service is provided for the given amount of time

    B = 1 - 1 / (R * C)
    A = outdoor_air_temperature / (R * C) + current_heating_output / C
    #     print(f'A={A} and B={B}')#, np.log(T_target-A/(1-B)) {np.log(T_target-A/(1-B))}')
    duration = -10

    T_in_boundary = calculate_max_temperature(outdoor_air_temperature,
                                              current_heating_output, R)

    if initial_indoor_air_temperature >= T_in_boundary:
      # decreasing heating output or keeping heating output constant
      #         print('Decrease heat output')
      if limit_indoor_air_temperature >= initial_indoor_air_temperature:
        duration = 0
      else:
        if T_in_boundary < limit_indoor_air_temperature:
          if T_in_boundary < initial_indoor_air_temperature:
            duration = np.log(
                (limit_indoor_air_temperature - A / (1 - B)) /
                (initial_indoor_air_temperature - A / (1 - B))) / np.log(B)
          else:
            duration = 0
        else:
          # the service can be provided for an infinite amount of time
          duration = -1
    else:
      #         print('Increase heat output')
      # increasing heating output
      if limit_indoor_air_temperature <= initial_indoor_air_temperature:
        # The limit temperature is below the initial indoor temperature
        duration = 0
      else:
        if T_in_boundary > limit_indoor_air_temperature:
          # The max indoor temperature is above the limit temperature thus time is not infinite.
          duration = np.log(
              (limit_indoor_air_temperature - A / (1 - B)) /
              (initial_indoor_air_temperature - A / (1 - B))) / np.log(B)
        else:
          duration = -1

    #     if duration == -1:
    #         print(f'The inside temperature of {T_limit}ºC will never be reached. The service can be provided for an infinite amount of time.')
    #     else:
    #         print(f'The service can be provided for {duration} sec.')

    if np.isnan(duration):
      print_text = f"There is an issue with the calculation of the duration.\n"
      print_text += (
          f"Dwelling with R={R}, "
          f"C={C}, T_in={initial_indoor_air_temperature}, "
          f"T_out={outdoor_air_temperature}, P_out={current_heating_output}, "
          f"T_limit={limit_indoor_air_temperature}, T_bound={T_in_boundary}.")
      raise ValueError(print_text)

  return round(duration, 3)
