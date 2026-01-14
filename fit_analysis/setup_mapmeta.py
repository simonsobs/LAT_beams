import os
import numpy as np
from sotodlib import core
import h5py
import yaml
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from astral import LocationInfo
from astral.sun import sun
from astropy.time import Time
from pathlib import Path
import argparse



def classify_path(path_str):
    parts = {p.lower() for p in Path(path_str).parts}

    if "mars" in parts:
        return "mars"
    elif "saturn" in parts:
        return "saturn"
    elif "taua" in parts:
        return "taua"
    else:
        return "other"

def get_obs_suntimes(ctime):

    """
    Helper function for the proccess of calculating suntimes
    """ 
    # General coordinates of obs site
    lat = -22.9606052
    lon = -67.7905487
    height = 5190  # meters
    
    dt_utc = datetime.fromtimestamp(ctime, tz=ZoneInfo("UTC"))
    loc = LocationInfo(name="Obs", region="Chile", timezone="America/Santiago",
                   latitude=lat, longitude=lon)
    
    
    s = sun(loc.observer, date=dt_utc, tzinfo=ZoneInfo("America/Santiago"))
    return s["sunrise"], s["sunset"], dt_utc

def classify_obs_time(start, end, t, transition_hours=1):
    """
    Classify a time as 'day', 'night', or 'transition' based on start/end times.

    Parameters
    ----------
    start : datetime
        Start time (e.g., sunrise)
    end : datetime
        End time (e.g., sunset)
    t : datetime
        Time to classify
    transition_hours : float
        Window before/after boundaries considered 'transition' (default 1 hour)

    Returns
    -------
    str
        'day', 'night', or 'transition'
    """
  
    if start < t < end:
        return 'day'
    else:
        return 'night'

parser = argparse.ArgumentParser()
parser.add_argument("cfg", help="Path to the config file")
args = parser.parse_args()

with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)

map_data_dir = cfg.get("map_data_dir", None)
context_path = cfg.get("context", None)
epochs = cfg.get("epochs", None)
map_ref = cfg.get("map_ref", None)

if map_data_dir is None:
    raise Exception("map_data_dir invalid in config file!")

if context_path is None:
    raise Exception("context invalid in config file!")

if epochs is None:
    print("No epochs specified in config, so no epoch tag will be created!")

if map_ref is None:
    map_ref = './'
    print("map_ref invalid in config file! Defaulting to local directory.")

context = core.Context(context_path)

files = []
for dirpath, dirnames, filenames in os.walk(map_data_dir):
    for name in filenames:
        file = os.path.join(dirpath, name)
        if "solved" in os.path.basename(file):
            files.append(file)

results = []
for file in files:
    basename = os.path.splitext(os.path.basename(file))[0]
    
    pattern = re.compile(
        r"""
        obs_(?P<obs_id>\d+)            # obs id
        _(?P<tube_slot>lati\d+)          # tube slot
        _(?P<wafers_present>[01]{3})   # wafer availability
        _(?P<wafer>ufm_[^_]+)        # wafer slot
        _(?P<band>f\d+)                # frequency
        """,
        re.VERBOSE
    )
    
    match = pattern.search(basename)
    result = match.groupdict()
    result['path_pattern'] = "_".join(os.path.splitext(file)[0].split('_')[:-1])
    obslist = context.obsdb.query(f"type=='obs' and subtype=='cal' and obs_id=='obs_{result['obs_id']}_{result['tube_slot']}_{result['wafers_present']}'")
    
    if len(obslist) > 1:
        raise Exception(f"More than one observation matches id obs_{result["obs_id"]}_{result["tube_slot"]}_{result["wafers_present"]}")
    
    obs = obslist[0]
    
    ctime = obs['timestamp']

    rise, set_, dt = get_obs_suntimes(ctime)
    # Observational time regime
    result['obstime_reg'] = classify_obs_time(rise, set_, dt)
    result['sunrise'] = rise.strftime("%Y-%m-%d %H:%M:%S")
    result['sunset'] = set_.strftime("%Y-%m-%d %H:%M:%S")
    result['date'] =  dt.strftime("%Y-%m-%d %H:%M:%S")
    result['ctime'] = ctime
    
    epoch = "N/A"
    if epochs:
        for ct_ranges in epochs:
            if ctime >= ct_ranges[1] and ctime < ct_ranges[2]:
                epoch = ct_ranges[0]
    
    if epoch == "N/A" and epochs:
        print(f"Warning! obs_{result["obs_id"]}_{result["tube_slot"]}_{result["wafers_present"]} is not within defined epochs! ({ctime})")
    
    result['epoch'] = epoch
    result['source'] = classify_path(result['path_pattern'])
    results.append(result)

with h5py.File(map_ref+"mapmeta.h5", "w") as f:
    for i, d in enumerate(results):
        gid = f"obs_{d['obs_id']}_{d['tube_slot']}_{d['wafers_present']}_{d['band']}_{d['wafer']}"
        g = f.create_group(gid)
        for key, val in d.items():
            if isinstance(val, np.ndarray):
                g.create_dataset(key, data=val, compression="gzip")
            else:
                g.attrs[key] = val