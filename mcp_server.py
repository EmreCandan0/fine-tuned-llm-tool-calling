import numpy as np
from fastmcp import FastMCP
from funcs_pool import get_geom_wkt_and_bounds, save_metadata, get_epsg_from_dataset, clear_temp_dirs
from osgeo import gdal
import os
from datetime import datetime
import time
import atexit

mcp = FastMCP(
    name="GeoTIFF Cropping Agent",
    instructions=""""
    This server analysis tiff or jp2 files crops them by the giving coordinates or instructions and converts it to the png
    Also it can calculate the mean NDVI and point NDVI according to given coordinates """,
    stateless_http=True
)

@atexit.register
def cleanup():
    clear_temp_dirs()

@mcp.tool()
def analyze_tiff(filepath: str) -> dict:
    """
    Gets the filepath from user and returns metadata/extent
    Tiff dosyasını analiz eder ve veritabanına kaydeder
    Çağırılma ifadeleri:
    'bu dosyayı analiz et','dosyayı veritabanına yaz',tiff işle ,'analiz ve kaydet','kayıt oluştur'
    """
    start_time = time.time()
    dataset = gdal.Open(filepath)
    metadata = dataset.GetMetadata()
    print("analyze_tiff called")
    source_path = os.path.abspath(filepath)
    key = "AREA_OR_POINT"
    value = metadata.get(key)
    file_size = round(os.path.getsize(source_path) / (1024 * 1024), 2)

    band_type_list = []
    for i in range(dataset.RasterCount):
        band = dataset.GetRasterBand(i + 1)
        color_interp_code = band.GetColorInterpretation()
        color_name = gdal.GetColorInterpretationName(color_interp_code).capitalize()
        if color_name:
            band_type_list.append(color_name)
    if set(band_type_list) == {"Red", "Green", "Blue"}:
        band_type = "RGB"
    elif "Gray" in band_type_list or len(band_type_list) == 1:
        band_type = "Panchromatic"
    else:
        band_type = ",".join(band_type_list)

    geom_str = get_geom_wkt_and_bounds(dataset)
    epsg_code = get_epsg_from_dataset(dataset)
    upload_time = datetime.now()

    save_metadata(
        os.path.basename(filepath),
        upload_time,
        epsg_code,
        value,
        geom_str,
        source_path,
        file_size,
        band_type
    )

    ds = gdal.Open(filepath)
    gt = ds.GetGeoTransform()
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + (ds.RasterXSize * gt[1])
    miny = maxy + (ds.RasterYSize * gt[5])

    elapsed_time = time.time() - start_time

    return {
        "message": "JP2 analyzed successfully.",
        "minx": minx,
        "miny": miny,
        "maxx": maxx,
        "maxy": maxy,
        "filename": os.path.basename(filepath),
        "elapsed_time": elapsed_time
    }
@mcp.tool()
def crop_image(filepath: str, minx: float, miny: float, maxx: float, maxy: float) -> dict:
    """
    Crops and saves a png file according to the given TIFF according to minx,miny,maxx,maxy values.
    If the reprojected TIFF does not exist, creates it.
    """

    output_path = f'./{os.path.splitext(os.path.basename(filepath))[0]}_cropped.png'


    ds = gdal.Open(filepath)
    arr = ds.GetRasterBand(1).ReadAsArray()
    min_val = arr.min()
    max_val = arr.max()
    print(minx, miny, maxx, maxy)
    options = gdal.TranslateOptions(
        format='PNG',
        projWin=[minx, maxy, maxx, miny],
        outputType=gdal.GDT_Byte,
        scaleParams=[[float(min_val), float(max_val), 0, 255]]

    )


    gdal.Translate(output_path, filepath, options=options)
    print("crop_image called")
    return {"image_url": output_path}

@mcp.tool()
def get_ndvi(filepath: str, x: float, y: float) -> dict:
    """
    Gets the filepath from user and x,y values then calculates the ndvi for the given x,y
    """
    dataset = gdal.Open(filepath)
    red_band = dataset.GetRasterBand(1).ReadAsArray().astype(float) #dem için 4 5
    nir_band = dataset.GetRasterBand(3).ReadAsArray().astype(float)

    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    ndvi_mean = ndvi.mean()

    gt = dataset.GetGeoTransform()
    px = int((x - gt[0]) / gt[1])
    py = int((y - gt[3]) / gt[5])
    print("get_ndvi called")
    if 0 <= px < ndvi.shape[1] and 0 <= py < ndvi.shape[0]:
        ndvi_point = ndvi[py, px]
    else:
        ndvi_point = None

    return {
        "ndvi_mean": ndvi_mean,
        "px" : px,
        "py" : py,
        "ndvi_point": ndvi_point
    }

@mcp.tool()
def get_dem(filepath: str, x: float, y: float) -> dict:
    dataset=gdal.Open(filepath)
    dem_band = dataset.GetRasterBand(1).ReadAsArray().astype(float)

    gt = dataset.GetGeoTransform()
    px = int((x - gt[0] )/ gt[1])
    py = int((y - gt[3]) / gt[5])

    if 0<=px<dem_band.shape[1] and 0<=py<dem_band.shape[0]:
        dem= dem_band[py,px]
    else:
        dem=None
    print("get_dem called")
    return {"Calculated DEM": dem}

if __name__ == "__main__":
    os.makedirs("temp", exist_ok=True)
    os.makedirs("static/outputs", exist_ok=True)
    mcp.run(
        transport="http",
         host="127.0.0.1",
        port=11436
 )


