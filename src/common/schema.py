from . import enums


class OutputDataSchema:
  DATEINDEX = "Datetime_UTC"
  OAT = "Temperature_(degreeC)"
  SOLARRADIATION = "Solar_radiation_(W/m2)"
  HDD = enums.Category.HDD.value
  CDD = enums.Category.CDD.value
  HEATINGSEASON = "Heating_season_flag"


class DataSchema:
  """Data schema for thermal model of a building."""
  IAT = "Average_indoor_air_temperature_(degreeC)"
  OAT = "Outdoor_air_temperature_(degreeC)"
  HEATINGOUTPUT = "Heating_output_(kW)"
  SOLARRADIATION = "Solar_radiation(W/m2)"
  SOLARGAINS = "Solar_gains_(kW)"
  OCCUPANCYGAINS = "Occupancy_gains_(kW)"
  APPLIANCESGAINS = "Appliances_gains_(kW)"
  TOTALGAINS = "Total_gains_(kW)"
  HEATINGSEASON = "Heating_season_flag"
  TIME_SECONDS = "Time_(s)"
  TIME_HOURS = "Time_(h)"


class ResultSchema:
  HEATINGDEMAND = "Heating demand (kWh)"
  COOLINGDEMAND = "Cooling demand (kWh)"
  YEAR = "Year"
  SPECIFICHEATINGDEMAND = "Specific heating demand (kWh/dwelling)"
  SPECIFICCOOLINGDEMAND = "Specific cooling demand (kWh/dwelling)"


class VisualisationSchema:
  IAT = "Average indoor air temperature\n(degreeC)"
  GAINS = "Heat gains (kW)"
