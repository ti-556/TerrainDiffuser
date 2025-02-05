import ee
import geemap
import argparse
import sys
import logging

def parse_arguments():
    """
    Parses command-line arguments provided by the user.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Process and export Sentinel-2 imagery and DEM data using Google Earth Engine.')
    parser.add_argument('--aoi', type=float, nargs=4, metavar=('lon_min', 'lat_min', 'lon_max', 'lat_max'),
                        help='Area of interest specified as longitude and latitude bounds: lon_min lat_min lon_max lat_max',
                        default=[-122.5, 37.0, -121.5, 38.0])
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                        help='Start date for the image collection (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2022-12-31',
                        help='End date for the image collection (YYYY-MM-DD)')
    parser.add_argument('--cloud-percentage', type=float, default=20,
                        help='Maximum cloud coverage percentage to filter images')
    parser.add_argument('--export-folder', type=str, default='EarthEngineExports',
                        help='Google Drive folder to export images to')
    parser.add_argument('--export-scale', type=float, default=10,
                        help='Scale in meters for the exported images')
    parser.add_argument('--export-name', type=str, default='terrain',
                        help='Exported file prefix')
    parser.add_argument('--map-name', type=str, default='map',
                        help='Interactive html map prefix')
    parser.add_argument('--project', type=str, default=None,
                        help='Google Cloud project ID (if required)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # authenticate and initialize Earth Engine
    try:
        ee.Authenticate()
        if args.project:
            ee.Initialize(project=args.project)
        else:
            ee.Initialize()
        logging.info('Successfully authenticated and initialized Earth Engine.')
    except Exception as e:
        logging.error('Failed to authenticate or initialize Earth Engine: %s', e)
        sys.exit(1)
        
    # define area of interest (AOI)
    geometry = ee.Geometry.Rectangle(args.aoi)
    logging.info('Area of interest set to: %s', args.aoi)
    
    # load and prepare DEM data
    dem = ee.Image('USGS/SRTMGL1_003').clip(geometry)
    logging.info('Loaded and clipped DEM data.')
    
    # load and prepare Sentinel-2 Level-2A data
    logging.info('Loading Sentinel-2 Level-2A data...')
    sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(geometry) \
        .filterDate(args.start_date, args.end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', args.cloud_percentage)) \
        .select(['B2', 'B3', 'B4', 'B8', 'SCL'])  # Include SCL for cloud masking
    logging.info('Sentinel-2 data loaded and filtered.')
    
    # cloud masking function for Level-2A data
    def maskS2clouds_L2A(image):
        scl = image.select('SCL')
        # Pixels classified as clouds or shadows
        mask = scl.neq(3).And(scl.neq(9)).And(scl.neq(8)).And(scl.neq(7))
        return image.updateMask(mask)
    
    # apply cloud masking
    sentinel2 = sentinel2.map(maskS2clouds_L2A)
    logging.info('Applied cloud masking to Sentinel-2 data.')
    
    # get the collection size on the client-side
    try:
        collection_size = sentinel2.size().getInfo()
        logging.info('Collection size: %d', collection_size)
    except Exception as e:
        logging.error('Failed to get collection size: %s', e)
        sys.exit(1)
    
    if collection_size > 0:
        # median reduction
        s2_median = sentinel2.median().clip(geometry)
        logging.info('Computed median composite of Sentinel-2 images.')
        
        # resample DEM to match Sentinel-2 resolution
        dem_resampled = dem.reproject(
            crs=s2_median.projection(),
            scale=args.export_scale
        )
        logging.info('Resampled DEM to match Sentinel-2 resolution.')
        
        # visualize w/ geemap
        center_lat = (args.aoi[1] + args.aoi[3]) / 2
        center_lon = (args.aoi[0] + args.aoi[2]) / 2
        Map = geemap.Map(center=[center_lat, center_lon], zoom=10)
        
        # add DEM layer to the map
        dem_vis_params = {
            'min': 0,
            'max': 3000,
            'palette': ['blue', 'green', 'yellow', 'orange', 'red']
        }
        Map.addLayer(dem_resampled, dem_vis_params, 'DEM')
        
        # add Sentinel-2 layer to the map
        s2_vis_params = {
            'bands': ['B4', 'B3', 'B2'],  # Red, Green, Blue
            'min': 0,
            'max': 3000
        }
        Map.addLayer(s2_median, s2_vis_params, 'Sentinel-2')
        
        # save the map to an html file
        Map.addLayerControl()
        map_file = args.map_name + '.html'
        try:
            Map.to_html(map_file)
            logging.info('Map saved to %s. Open it in a web browser to view the map.', map_file)
        except Exception as e:
            logging.error('Failed to save map to HTML file: %s', e)
        
        # export DEM
        try:
            dem_export = ee.batch.Export.image.toDrive(
                image=dem_resampled,
                description=args.export_name + 'DEM',
                folder=args.export_folder,
                fileNamePrefix=args.export_name + 'DEM',
                scale=args.export_scale,
                region=geometry.getInfo()['coordinates'],
                crs=s2_median.projection(),
                maxPixels=1e13
            )
            dem_export.start()
            logging.info('DEM export task started.')
        except Exception as e:
            logging.error('Failed to start DEM export task: %s', e)
        
        # export Sentinel-2 imagery
        try:
            s2_export = ee.batch.Export.image.toDrive(
                image=s2_median.select(['B2', 'B3', 'B4']),  # Ensure only selected bands are exported
                description=args.export_name + 'S2',
                folder=args.export_folder,
                fileNamePrefix=args.export_name + 'S2',
                scale=args.export_scale,
                region=geometry.getInfo()['coordinates'],
                crs=s2_median.projection(),
                maxPixels=1e13
            )
            s2_export.start()
            logging.info('Sentinel-2 export task started.')
        except Exception as e:
            logging.error('Failed to start Sentinel-2 export task: %s', e)
        
        logging.info('Export tasks started. Check your Earth Engine Tasks tab or your Google Drive.')
    else:
        logging.warning('The ImageCollection is empty. Adjust your filters and try again.')

if __name__ == '__main__':
    main()