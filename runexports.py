import yaml
import subprocess

# AoI yaml file
yaml_file = 'aoilist.yaml'

# Parameters
project_name = 'xxxxxxxxxxx' # google earth engine project name
script_path = './demtextureexporter.py'  # exporter script

with open(yaml_file, 'r') as f:
    data = yaml.safe_load(f)

areas = data.get('areas', [])

for i, area in enumerate(areas):
    name = area['name']
    aoi = area['aoi']
    aoi_args = [str(coord) for coord in aoi]

    export_name = f"image{i}"
    map_name = f"{name}.html"

    cmd = [
        'python', script_path,
        '--export-name', export_name,
        '--map-name', map_name,
        '--aoi', aoi_args[0], aoi_args[1], aoi_args[2], aoi_args[3],
        '--project', project_name
    ]

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
