from enum import Enum, auto


class DataSource(Enum):
  IES = "IES"
  MEASURED = "Measured"
  RC = "R1C1"


class Category(Enum):
  """Category of degree days used"""
  HDD = "hdd"
  CDD = "cdd"


class ExtractFile(Enum):
  TEMPERATURE = "t2m", "2m_temperature"
  SOLARRADIATION = "ssr", "surface_net_solar_radiation"

  @property
  def filename_key(self) -> str:
    """Get the str which is used in naming the file."""
    return self.value[1]

  @property
  def column_name(self) -> str:
    """Get the column name related to the key."""
    return self.value[0]


class Area(Enum):

  CARDIFF = "Cardiff"
  NY = "NY"


# class TimeStep(Enum):
#     HALFHOUR = 30, UnitRegistry().minutes
#     HOUR = 60, UnitRegistry().minutes

#     def __init__(self, magnitude: float, units: UnitRegistry):
#         self.magnitude: float = magnitude
#         self.units: UnitRegistry = units
