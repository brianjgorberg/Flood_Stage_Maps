#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:
from pathlib import Path
import csv

import numpy as np
from osgeo import gdal, ogr
gdal.UseExceptions()

def define_projection_ASCII(
    ascii_path,
    epsg=26904,
):
    """
    Create a .prj file next to an ESRI ASCII raster.
    """

    ascii_path = Path(ascii_path)

    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(epsg)

    prj_path = ascii_path.with_suffix(".prj")

    with open(
        prj_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            spatial_reference.ExportToWkt()
        )

    print(
        f"Projection file created: {prj_path}"
    )

    return prj_path


def convert_ASCII_to_GeoTIFF(
    ascii_path,
    output_tif_path=None,
):
    """
    Convert an ESRI ASCII grid (.asc) to a GeoTIFF (.tif).

    Parameters
    ----------
    ascii_path : str or Path
        Path to the input ASCII raster.

    output_tif_path : str or Path, optional
        Path for the output GeoTIFF.
        If None, the GeoTIFF is saved beside the ASCII file
        using the same filename with a .tif extension.

    Returns
    -------
    Path
        Path to the created GeoTIFF.
    """

    ascii_path = Path(ascii_path)

    if output_tif_path is None:
        output_tif_path = ascii_path.with_suffix(".tif")
    else:
        output_tif_path = Path(output_tif_path)

    if not ascii_path.exists():
        raise FileNotFoundError(
            f"ASCII file not found: {ascii_path}"
        )

    input_raster = gdal.Open(
        str(ascii_path)
    )

    if input_raster is None:
        raise RuntimeError(
            f"GDAL could not open: {ascii_path}"
        )

    gdal.Translate(
        str(output_tif_path),
        input_raster,
        format="GTiff",
        creationOptions=[
            "COMPRESS=DEFLATE",
        ],
    )

    input_raster = None

    print(
        f"GeoTIFF created: {output_tif_path}"
    )

    return output_tif_path
def zonal_mean_from_shapefile(
    raster_path,
    shapefile_path,
    zone_field,
    output_field,
):
    """
    Calculate the mean raster value within each polygon feature
    using ArcGIS-style cell-center zonal statistics.

    The polygon is rasterized using the exact cell size,
    alignment, and spatial reference of the input raster.

    Only cells whose centers fall inside the polygon are included.
    NoData raster cells are ignored.

    The calculated mean is written directly into the shapefile
    attribute table.

    Parameters
    ----------
    raster_path : str or Path
        Input value raster (.tif).

    shapefile_path : str or Path
        Input polygon shapefile.

    zone_field : str
        Unique ID field defining each polygon zone.

    output_field : str
        Name of the field where the mean raster value will be stored.

        Note:
        Shapefile field names are limited to 10 characters.

    Returns
    -------
    None
    """

    from pathlib import Path

    import numpy as np
    from osgeo import gdal, ogr

    gdal.UseExceptions()

    raster_path = Path(raster_path)
    shapefile_path = Path(shapefile_path)

    # --------------------------------------------------
    # Check inputs
    # --------------------------------------------------
    if not raster_path.exists():
        raise FileNotFoundError(
            f"Raster not found: {raster_path}"
        )

    if not shapefile_path.exists():
        raise FileNotFoundError(
            f"Shapefile not found: {shapefile_path}"
        )

    if len(output_field) > 10:
        raise ValueError(
            "Shapefile field names cannot exceed 10 characters. "
            f"Received: '{output_field}'"
        )

    # --------------------------------------------------
    # Open raster
    # --------------------------------------------------
    raster_ds = gdal.Open(
        str(raster_path),
        gdal.GA_ReadOnly,
    )

    if raster_ds is None:
        raise RuntimeError(
            f"GDAL could not open raster: {raster_path}"
        )

    raster_band = raster_ds.GetRasterBand(1)

    geotransform = raster_ds.GetGeoTransform()
    raster_projection = raster_ds.GetProjection()

    raster_cols = raster_ds.RasterXSize
    raster_rows = raster_ds.RasterYSize

    nodata = raster_band.GetNoDataValue()

    # This function assumes a normal north-up raster.
    if (
        geotransform[2] != 0
        or geotransform[4] != 0
    ):
        raise ValueError(
            "Rotated rasters are not supported by this function."
        )

    pixel_width = geotransform[1]
    pixel_height = abs(
        geotransform[5]
    )

    # --------------------------------------------------
    # Open shapefile in UPDATE mode
    # --------------------------------------------------
    vector_ds = ogr.Open(
        str(shapefile_path),
        1,
    )

    if vector_ds is None:
        raise RuntimeError(
            f"GDAL could not open shapefile for editing: "
            f"{shapefile_path}"
        )

    layer = vector_ds.GetLayer()

    # --------------------------------------------------
    # Verify zone field exists
    # --------------------------------------------------
    layer_definition = layer.GetLayerDefn()

    existing_fields = [
        layer_definition
        .GetFieldDefn(i)
        .GetName()
        for i in range(
            layer_definition.GetFieldCount()
        )
    ]

    if zone_field not in existing_fields:
        raise ValueError(
            f"Zone field '{zone_field}' does not exist "
            f"in {shapefile_path.name}"
        )

    # --------------------------------------------------
    # Create output field if it does not exist
    # --------------------------------------------------
    if output_field not in existing_fields:

        new_field = ogr.FieldDefn(
            output_field,
            ogr.OFTReal,
        )

        new_field.SetWidth(18)
        new_field.SetPrecision(4)

        result = layer.CreateField(
            new_field
        )

        if result != 0:
            raise RuntimeError(
                f"Could not create field: {output_field}"
            )

    # Refresh definition after adding field
    layer_definition = layer.GetLayerDefn()

    output_field_index = (
        layer_definition.GetFieldIndex(
            output_field
        )
    )

    # --------------------------------------------------
    # Process each polygon
    # --------------------------------------------------
    layer.ResetReading()

    processed_count = 0
    empty_count = 0

    for feature in layer:

        zone_id = feature.GetField(
            zone_field
        )

        geometry = feature.GetGeometryRef()

        if geometry is None:
            continue

        # Polygon bounding box
        xmin, xmax, ymin, ymax = (
            geometry.GetEnvelope()
        )

        # --------------------------------------------------
        # Determine raster window covering polygon
        # --------------------------------------------------
        x_offset = int(
            np.floor(
                (
                    xmin
                    - geotransform[0]
                )
                / pixel_width
            )
        )

        y_offset = int(
            np.floor(
                (
                    geotransform[3]
                    - ymax
                )
                / pixel_height
            )
        )

        x_end = int(
            np.ceil(
                (
                    xmax
                    - geotransform[0]
                )
                / pixel_width
            )
        )

        y_end = int(
            np.ceil(
                (
                    geotransform[3]
                    - ymin
                )
                / pixel_height
            )
        )

        # Clip window to raster boundaries
        x_offset = max(
            0,
            x_offset,
        )

        y_offset = max(
            0,
            y_offset,
        )

        x_end = min(
            raster_cols,
            x_end,
        )

        y_end = min(
            raster_rows,
            y_end,
        )

        x_size = (
            x_end - x_offset
        )

        y_size = (
            y_end - y_offset
        )

        # Polygon does not intersect raster
        if (
            x_size <= 0
            or y_size <= 0
        ):

            feature.UnsetField(
                output_field_index
            )

            layer.SetFeature(
                feature
            )

            empty_count += 1
            continue

        # --------------------------------------------------
        # Read raster values for this polygon window
        # --------------------------------------------------
        raster_values = (
            raster_band.ReadAsArray(
                x_offset,
                y_offset,
                x_size,
                y_size,
            )
        )

        if raster_values is None:
            continue

        # --------------------------------------------------
        # Build temporary mask raster
        #
        # IMPORTANT:
        # ALL_TOUCHED is NOT enabled.
        #
        # This keeps the normal GDAL rasterization behavior,
        # which uses the cell-center rule and therefore
        # matches ArcGIS polygon-zone rasterization.
        # --------------------------------------------------
        mask_ds = (
            gdal.GetDriverByName(
                "MEM"
            )
            .Create(
                "",
                x_size,
                y_size,
                1,
                gdal.GDT_Byte,
            )
        )

        mask_geotransform = (
            geotransform[0]
            + x_offset
            * pixel_width,
            pixel_width,
            0.0,
            geotransform[3]
            - y_offset
            * pixel_height,
            0.0,
            -pixel_height,
        )

        mask_ds.SetGeoTransform(
            mask_geotransform
        )

        mask_ds.SetProjection(
            raster_projection
        )

        # --------------------------------------------------
        # Make temporary vector containing only
        # the current polygon
        # --------------------------------------------------
        memory_driver = (
            ogr.GetDriverByName(
                "Memory"
            )
        )

        temp_vector = (
            memory_driver.CreateDataSource(
                ""
            )
        )

        temp_layer = (
            temp_vector.CreateLayer(
                "zone",
                srs=layer.GetSpatialRef(),
                geom_type=geometry.GetGeometryType(),
            )
        )

        temp_feature = ogr.Feature(
            temp_layer.GetLayerDefn()
        )

        temp_feature.SetGeometry(
            geometry.Clone()
        )

        temp_layer.CreateFeature(
            temp_feature
        )

        temp_feature = None

        # --------------------------------------------------
        # Rasterize polygon
        #
        # Default = cell-center inclusion.
        # DO NOT add ALL_TOUCHED=TRUE if you want
        # ArcGIS Zonal Statistics behavior.
        # --------------------------------------------------
        gdal.RasterizeLayer(
            mask_ds,
            [1],
            temp_layer,
            burn_values=[1],
        )

        mask = (
            mask_ds
            .GetRasterBand(1)
            .ReadAsArray()
        )

        # --------------------------------------------------
        # Select valid raster cells
        # --------------------------------------------------
        valid = (
            mask == 1
        )

        # ArcGIS default behavior:
        # ignore NoData cells inside a zone
        if nodata is not None:

            if np.isnan(nodata):

                valid &= ~np.isnan(
                    raster_values
                )

            else:

                valid &= (
                    raster_values
                    != nodata
                )

        valid &= np.isfinite(
            raster_values
        )

        values_inside = (
            raster_values[
                valid
            ]
        )

        # --------------------------------------------------
        # Calculate mean and write directly
        # into shapefile
        # --------------------------------------------------
        if values_inside.size == 0:

            # ArcGIS can produce no statistic when
            # no raster-cell centers fall inside a zone.
            feature.UnsetField(
                output_field_index
            )

            empty_count += 1

        else:

            mean_value = float(
                np.mean(
                    values_inside
                )
            )

            feature.SetField(
                output_field,
                mean_value,
            )

            processed_count += 1

        layer.SetFeature(
            feature
        )

        # Clean temporary objects
        mask_ds = None
        temp_layer = None
        temp_vector = None

    # --------------------------------------------------
    # Save / close
    # --------------------------------------------------
    layer.SyncToDisk()

    layer = None
    vector_ds = None

    raster_band = None
    raster_ds = None

    print(
        f"Zonal mean complete.\n"
        f"Raster: {raster_path.name}\n"
        f"Shapefile: {shapefile_path.name}\n"
        f"Output field: {output_field}\n"
        f"Zones calculated: {processed_count}\n"
        f"Zones with no included raster cells: "
        f"{empty_count}"
    )

def read_shapefile_to_dataframe(
    shapefile_path,
):
    """
    Read the attribute table from an OGR-readable vector dataset.

    This function does NOT require pandas.

    Parameters
    ----------
    shapefile_path : str or Path
        Path to the shapefile or OGR-readable vector dataset.

    Returns
    -------
    list of dict
        One dictionary per feature containing the attribute fields.
    """

    from pathlib import Path
    from osgeo import ogr

    shapefile_path = Path(
        shapefile_path
    )

    # Open vector dataset
    vector_ds = ogr.Open(
        str(shapefile_path)
    )

    if vector_ds is None:
        raise FileNotFoundError(
            f"Could not open vector dataset: "
            f"{shapefile_path}"
        )

    # Get first layer
    layer = vector_ds.GetLayer()

    if layer is None:
        raise RuntimeError(
            f"No vector layer found in: "
            f"{shapefile_path}"
        )

    # Get field names
    layer_definition = (
        layer.GetLayerDefn()
    )

    field_names = [
        layer_definition
        .GetFieldDefn(i)
        .GetName()
        for i in range(
            layer_definition.GetFieldCount()
        )
    ]

    # Read attributes
    rows = []

    for feature in layer:

        row = {}

        for field_name in field_names:

            row[field_name] = (
                feature.GetField(
                    field_name
                )
            )

        rows.append(
            row
        )

    # Close dataset
    layer = None
    vector_ds = None

    return rows

# In[ ]: