#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:
from pathlib import Path
import sys

import gdal_functions as gdf


command = sys.argv[1]
input_file = Path(sys.argv[2])

if command == "define_projection_ASCII":

    gdf.define_projection_ASCII(
        ascii_path=input_file,
        epsg=26904,
    )

elif command == "convert_ASCII_to_GeoTIFF":

    gdf.convert_ASCII_to_GeoTIFF(
        ascii_path=input_file,
    )
    
elif command == "zonal_mean_from_shapefile":

    raster_path = Path(
        sys.argv[2]
    )

    shapefile_path = Path(
        sys.argv[3]
    )

    zone_field = sys.argv[4]

    output_field = sys.argv[5]

    gdf.zonal_mean_from_shapefile(
        raster_path=raster_path,
        shapefile_path=shapefile_path,
        zone_field=zone_field,
        output_field=output_field,
    )

elif command == "read_shapefile_to_dataframe":

    import json

    shapefile_path = Path(
        sys.argv[2]
    )

    rows = gdf.read_shapefile_to_dataframe(
        shapefile_path=shapefile_path,
    )

    print(
        json.dumps(
            rows
        )
    )

else:
    raise ValueError(f"Unknown command: {command}")

# In[ ]: