from . import enums


class geoLookupSchema:
  """Schema for England and Wales geocodes"""
  postcode = "pcds"
  oa = "oa11cd"
  lsoa = "lsoa11cd"
  msoa = "msoa11cd"
  ladcd = "ladcd"
  ladnm = "ladnm"


class DwellingDataSchema:
  THERMAL_LOSSES = "Average thermal losses kW/K"
  THERMAL_CAPACITY = "Average thermal capacity kJ/K"
  FLOOR_AREA = "Average floor area m2"
  CIBSE_CITY = "CIBSE_city"
  REGION = "Region"
  LOCAL_AUTHORITY = "Local Authority"
  THERMAL_CAPACITY_LEVEL = "Thermal capacity level"
  NB_DWELLINGS = "Number of dwellings"
  COOLING_DEMAND = "Cooling demand (kWh)"
  LSOA = "LSOA_code"
  LADCD = "ladcd"
  LADNM = "ladnm"
  DWELLING_FORMS = "Dwelling forms"
  HEATING_SYSTEMS = "Heating systems"


class ValidationDataSchema:
  SOLARRADIATION = 'Global Radiation (W/m2)'
  OAT = 'Dry Bulb (degC)'


class OutputDataSchema:
  DATEINDEX = "Datetime_UTC"
  OAT = "Temperature_(degreeC)"
  SOLARRADIATION = "Solar_radiation_(W/m2)"
  HDD = enums.Category.HDD.value
  CDD = enums.Category.CDD.value
  HEATINGSEASON = "Heating_season_flag"


class DataSchema:
  """Data schema for thermal model of a building."""
  SOLAR_DECLINATION = 'Solar declination'
  IAT = "Average_indoor_air_temperature_(degreeC)"
  OAT = "Outdoor_air_temperature_(degreeC)"
  HEATINGOUTPUT = "Heating_output_(kW)"
  SOLARRADIATION = "Solar_radiation(W/m2)"
  SOLARGAINS = "Solar_gains_(kW)"
  OCCUPANCYGAINS = "Occupancy_gains_(kW)"
  APPLIANCESGAINS = "Appliances_gains_(kW)"
  VENTILATION = "Ventilation_losses_(kW)"
  IHG = "Internal_heat_gains_(kW)"
  TOTALGAINS = "Total_gains_(kW)"
  HEATINGSEASON = "Heating_season_flag"
  TIME_SECONDS = "Time_(s)"
  TIME_HOURS = "Time_(h)"
  DATETIME = "Datetime_UTC"


class ResultSchema:
  HEATINGDEMAND = "Heating demand (kWh)"
  COOLINGDEMAND = "Cooling demand (kWh)"
  YEAR = "Year"
  SPECIFICHEATINGDEMAND_DWELLING = "Specific heating demand (kWh/dwelling)"
  SPECIFICCOOLINGDEMAND_DWELLING = "Specific cooling demand (kWh/dwelling)"
  SPECIFICCOOLINGDEMAND_AREA = "Specific cooling demand (kWh/m2)"
  INDEX = "Index"
  LSOA = DwellingDataSchema.LSOA


class VisualisationSchema:
  IAT = "Average indoor\nair temperature ($^\circ$C)"
  HOURLY_OAT = "Hourly outdoor\nair temperature ($^\circ$C)"
  GAINS = "Heat gains (kW)"
  PEAK_COOLING = "Peak cooling demand (kW)"
  COOLINGDEMAND = ResultSchema.COOLINGDEMAND
  DEMAND_HEADROOM = "Thermal demand headroom (kW)"
  DEMAND_HEADROOM_AFTER_COOLING = "Thermal demand headroom after cooling demand (kW)"
  SPECIFICCOOLINGDEMAND_DWELLING = "Specific cooling demand\n(kWh/dwelling)"
  SPECIFICCOOLINGDEMAND_AREA = "Specific cooling demand\n(kWh/m2)"
