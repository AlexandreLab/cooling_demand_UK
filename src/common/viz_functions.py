import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from dotenv import load_dotenv
from mpl_toolkits.axes_grid1 import make_axes_locatable

load_dotenv()  # take environment variables from .env.


def get_UK_LSOA_map() -> gpd:
  path_map = Path(os.getenv('PATH_ONEDRIVE')) / r'General/resources/maps'
  file = "UK_2011_Census_boundaries_LSOA_fixed_v2.geojson"
  path_map = path_map / file
  map_df = get_map(path_map)
  filt = map_df['LSOA11CD'].isna()
  map_df.loc[filt, 'LSOA11CD'] = map_df.loc[filt, 'DataZone']
  return map_df


def get_map(path_map: Path) -> gpd:

  map_df = gpd.read_file(path_map)
  return map_df
  # filt = map_df['GeographyCode'].isin(dfes_dataf.index)
  # map_df.head()


def plot_map(map_df: gpd,
             target: str,
             ax,
             vmin: float = None,
             vmax: float = None,
             cmap: float = None,
             legend: bool = False,
             label_legend: str = '',
             **kwargs):

  # map_df = map_df.dropna(subset=[target])
  # create figure and axes for Matplotlib
  if vmin is None:
    if map_df[target].min() < 0:
      vmin = map_df[target].min()
    else:
      vmin = 0
  if vmax is None:
    vmax = map_df[target].max()

  if cmap == None:
    cmap = 'Blues'

  ax.axis('off')
  ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

  map_df = map_df.to_crs(epsg=3395)  # mercator projections

  if legend:
    legend_kwds = {
        "label": label_legend,
        "orientation": "vertical",
        "shrink": .2,
        "format": tkr.FuncFormatter(lambda x, p: "{:,.0f}".format(x))
    }
    legend_kwds.update(kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    final_map = map_df.plot(
        column=target,
        cmap=cmap,
        linewidth=0,  #0.01,
        ax=ax,
        edgecolor='black',
        vmin=vmin,
        vmax=vmax,
        legend=legend,
        legend_kwds=legend_kwds,
        cax=cax)
  else:
    final_map = map_df.plot(column=target,
                            cmap=cmap,
                            linewidth=0.01,
                            ax=ax,
                            edgecolor='black',
                            vmin=vmin,
                            vmax=vmax)
  plt.close()
  return final_map
  return final_map
