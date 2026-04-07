"""write_context.py -- create context.yaml & support databases for SSO sims

This script indexes output in TOAST's native HDF5 format.  You can
index any number of output directories and it will figure out what's
up with the tubes and telescopes and so on.

Invoke like this:

  python write_context.py out_f030_w42_Jupiter out_f150_i1_Jupiter ...

(Note the HDF data are expected to be in "data" subdir of each passed
argument.)

"""

import glob
import os
import sys
import time

import h5py
import numpy as np
import sotodlib.toast as sotoast
import yaml
from so3g.proj import quat
from sotodlib.core import metadata
from sotodlib.io.metadata import read_dataset, write_dataset

DEG = np.pi / 180

WAFER_CACHE = {}


def get_wafer_info(telescope, cache_file=None):
    if telescope not in WAFER_CACHE and cache_file is not None:
        try:
            WAFER_CACHE[telescope] = read_dataset(cache_file, telescope)
        except:
            print("Failed to load from cache file.")
    if telescope not in WAFER_CACHE:
        focalplane = sotoast.SOFocalplane(telescope=telescope)
        # Reduced info ...
        subkeys = ["band", "tube_slot", "wafer_slot"]
        subitems = set()
        for k in focalplane.keys():
            subitems.add(tuple([focalplane[k][_k] for _k in subkeys]))
        wafers = metadata.ResultSet(keys=subkeys)
        wafers.rows = sorted(list(subitems))
        WAFER_CACHE[telescope] = wafers
        if cache_file is not None:
            write_dataset(wafers, cache_file, telescope)
    return WAFER_CACHE[telescope]


tube_types = {
    "f030": "LF",
    "f040": "LF",
    "f090": "MF",
    "f150": "MF",
    "f230": "UHF",
    "f290": "UHF",
}


def guess_tube(telescope, wafers):
    if telescope == "SAT":
        for t in ["SAT1", "SAT2", "SAT3", "SAT4"]:
            try:
                return guess_tube(t, wafers)
            except AssertionError:
                pass
        assert False  # Did not found wafer in any SAT?

    info = get_wafer_info(telescope, "telcache")
    s = np.zeros(len(info), bool)
    for w in wafers:
        s += info["wafer_slot"] == w

    # Determine the tube_slot ...
    tube_info = info.subset(keys=["tube_slot"], rows=s).distinct()
    assert len(set(tube_info["tube_slot"])) == 1  # only one tube_slot!
    tube_slot = tube_info["tube_slot"][0]

    # Restrict to this tube only to get the slot_mask.
    info = info.subset(
        keys=["wafer_slot"], rows=info["tube_slot"] == tube_slot
    ).distinct()
    wafer_list = sorted(list(info["wafer_slot"]))

    slot_mask = ""
    for w in wafer_list:
        slot_mask += "1" if w in wafers else "0"

    return telescope, tube_slot, slot_mask, wafer_list


def extract_detdb(hg, db=None):
    if db is None:
        db = metadata.DetDb()
        db.create_table(
            "base",
            [
                "`det_id_` text",  # we can't use "det_id"; rename later
                "`readout_id` text",
                "`wafer_slot` text",
                "`special_ID` text",
                "`tel_type` text",
                "`tube_type` text",
                "`band` text",
                "`fcode` text",
                "`toast_band` text",
            ],
        )
        db.create_table(
            "quat",
            [
                "`r` float",
                "`i` float",
                "`j` float",
                "`k` float",
            ],
        )

    existing = list(db.dets()["name"])

    tel_type = hg["instrument"].attrs.get("telescope_name")
    if tel_type in ["LAT", "SAT"]:
        pass
    else:
        # new toast has TELn_TUBE
        tel_type = tel_type.split("_")[0][:3]
    assert tel_type in ["LAT", "SAT"]

    fp = hg["instrument"]["focalplane"]
    for dv in fp:
        v = {_k: dv[_k].decode("ascii") for _k in ["wafer_slot", "band", "name"]}
        k = v.pop("name")
        if k in existing:
            continue
        v["special_ID"] = int(dv["uid"])
        v["toast_band"] = v["band"]
        v["band"] = v["toast_band"].split("_")[1]
        v["fcode"] = v["band"]
        v["tel_type"] = tel_type
        v["tube_type"] = tube_types[v["band"]]
        v["det_id_"] = "DET_" + k
        v["readout_id"] = k
        db.add_props("base", k, **v, commit=False)
        db.add_props(
            "quat",
            k,
            **{
                "r": dv["quat"][3],
                "i": dv["quat"][0],
                "j": dv["quat"][1],
                "k": dv["quat"][2],
            },
        )

    db.conn.commit()
    db.validate()
    return db


def extract_obs_info(h):
    t = np.asarray(h["shared"]["times"])[[0, -1]]
    az = np.asarray(h["shared"]["azimuth"][()])
    el = np.asarray(h["shared"]["elevation"][()])
    el_nom = (el.max() + el.min()) / 2
    el_span = el.max() - el.min()
    # Put az in a single branch ...
    az_cut = az[0] - np.pi
    az = (az - az_cut) % (2 * np.pi) + az_cut
    az_span = az.max() - az.min()
    az_nom = (az.max() + az.min()) / 2 % (2 * np.pi)

    data = {
        "toast_obs_name": h.attrs["observation_name"],
        "toast_obs_uid": int(h.attrs["observation_uid"]),
        "target": h.attrs["observation_name"].split("-")[0].lower(),
        "start_time": t[0],
        "stop_time": t[1],
        "timestamp": t[0],
        "duration": t[1] - t[0],
        "type": "obs",
        "subtype": "survey",
        "el_nom": el_nom / DEG,
        "el_span": el_span / DEG,
        "az_nom": az_nom / DEG,
        "az_span": az_span / DEG,
        "roll_nom": 0.0,
        "roll_span": 0.0,
    }
    return data


def detdb_to_focalplane(db):
    # Focalplane compatible with, like, planet mapper.
    fp = metadata.ResultSet(keys=["dets:readout_id", "xi", "eta", "gamma"])
    for row in db.props(
        props=["readout_id", "quat.r", "quat.i", "quat.j", "quat.k"]
    ).rows:
        q = quat.quat(*row[1:])
        xi, eta, gamma = quat.decompose_xieta(q)
        fp.rows.append((row[0], xi, eta, (gamma) % (2 * np.pi)))
    return fp


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--context-dir", default="context/")
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Work around for relative paths bug in sotodlib ...",
    )
    parser.add_argument(
        "--test", action="store_true", help="Only process 5 items then exit."
    )
    parser.add_argument(
        "--dichroic-sub",
        nargs=2,
        help="Tack on second band by substitution, e.g. f090 f150",
    )
    parser.add_argument(
        "export_dirs", nargs="+", help="Directories to search for HDF data files."
    )

    args = parser.parse_args()

    if not os.path.exists(args.context_dir):
        os.makedirs(args.context_dir, exist_ok=True)

    tel_info_cachefile = os.path.join(args.context_dir, "tels.h5")

    obsfiledb = metadata.ObsFileDb()
    obsdb = metadata.ObsDb()
    obsdb.add_obs_columns(
        [
            ## Standardized
            "timestamp float",
            "duration float",
            "start_time float",
            "stop_time float",
            "type string",
            "subtype string",
            "telescope string",
            "telescope_flavor string",
            "tube_slot string",
            "tube_flavor string",
            "detector_flavor string",
            ## Standardizing soon
            "wafer_slot_mask string",
            "el_nom float",
            "el_span float",
            "az_nom float",
            "az_span float",
            "roll_nom float",
            "roll_span float",
            ## Extensions
            "wafer_slots string",
            "type string",
            "target string",
            "toast_obs_name string",
            "toast_obs_uid string",
        ]
    )

    detsets = {}
    item_count = 0

    # As we process data directories, data files from other data dirs
    # can get pulled into coherent observations; record them here to
    # not duplicate.
    handled = []

    # Loop over data dirs...
    for export_dir in args.export_dirs:
        files = glob.glob(os.path.join(export_dir, "*h5"))
        print(f"Found {len(files)} files in {export_dir}")

        for filename in files:
            if any([os.path.samefile(filename, f) for f in handled]):
                print(f" -- skipping {filename} -- already bundled")
                continue

            with h5py.File(filename, "r") as h:
                detdb = extract_detdb(h, db=None)
                obs_info = extract_obs_info(h)

            # This will be one band, one wafer.
            props = detdb.props()
            tel_type = props["tel_type"][0]
            wafers = set(props["wafer_slot"])

            if args.dichroic_sub:
                assert len(wafers) == 1

            telescope, tube, slot_mask, all_wafers = guess_tube(tel_type, wafers)
            bands = list(set(props["band"]))

            base_wafer = list(wafers)[0]
            base_band = bands[0]
            if args.dichroic_sub:
                all_bands = args.dichroic_sub
            else:
                all_bands = [base_band]

            wafers_found = []
            for wafer in all_wafers:
                both = True
                for band in all_bands:
                    filename_d = filename.replace(base_band, band).replace(
                        base_wafer, wafer
                    )
                    if not os.path.exists(filename_d):
                        both = False
                        continue
                    if filename_d == filename:  # already loaded...
                        continue
                    with h5py.File(filename_d, "r") as h:
                        detdb_d = extract_detdb(h, db=None)
                        obs_info_d = extract_obs_info(h)
                    for _n, _p in zip(detdb_d.dets(), detdb_d.props()):
                        _p1 = {k: v for k, v in _p.items() if not k.startswith("quat.")}
                        _p2 = {k[5:]: v for k, v in _p.items() if k.startswith("quat.")}
                        detdb.add_props("base", _n["name"], **_p1)
                        detdb.add_props("quat", _n["name"], **_p2)
                if both:
                    print(f" -- including {wafer} -- bands {all_bands}")
                    wafers_found.append(wafer)

            # Convert detdb to ResultSet
            props = detdb.props()
            props.keys[props.keys.index("det_id_")] = "det_id"  # hotwire

            # Merge in that telescope name
            props.merge(
                metadata.ResultSet(keys=["telescope"], src=[(telescope,)] * len(props))
            )

            obs_info.update(
                {
                    "telescope": telescope,
                    "telescope_flavor": telescope[:3],
                    "detector_flavor": "TES",
                    "tube_slot": tube,
                    "tube_flavor": props["tube_type"][0],
                    "wafer_slot_mask": "_" + slot_mask,
                    "wafer_slots": ",".join(wafers),
                }
            )

            if args.dichroic_sub:
                # In this format, there is one file per wafer and per
                # band.  There should be one detset per wafer, and the
                # loader will handle the rest.
                slot_mask = "".join([str(int(w in wafers_found)) for w in all_wafers])
                obs_id = f'{int(obs_info["timestamp"])}_{tube}_{slot_mask}'

                for w in wafers_found:
                    detset = w
                    det_select = detdb.props()["wafer_slot"] == w

                    if detset not in detsets:
                        fp = detdb_to_focalplane(detdb)
                        detsets[detset] = [
                            props.subset(rows=det_select),
                            fp.subset(rows=det_select),
                        ]
                        obsfiledb.add_detset(detset, props["readout_id"][det_select])

                    from_here = filename.replace(base_wafer, w)

                    # Handle referencing the file from where context.yaml will live
                    path = os.path.split(from_here)[0]
                    practical_path = path
                    if not path.startswith("/"):
                        if args.context_dir.startswith("/"):
                            practical_path = os.path.abspath(path)
                        else:
                            practical_path = os.path.relpath(path, args.context_dir)

                    wafer_file = os.path.join(
                        practical_path, os.path.split(from_here)[1]
                    )
                    obsfiledb.add_obsfile(wafer_file, obs_id, detset, 0, 1)

                    for b in all_bands:
                        handled.append(from_here.replace(base_band, b))

            else:

                # In this format, all dets for the set of wafers and a
                # single band are stored in one file.  Create that detset
                # name from the list of wafers + band(s).
                detset = "_".join(sorted(list(wafers)) + sorted(list(bands)))
                if detset not in detsets:
                    fp = detdb_to_focalplane(detdb)
                    detsets[detset] = [props, fp]
                    obsfiledb.add_detset(detset, props["readout_id"])

                obs_id = f'{int(obs_info["timestamp"])}_{tube}_{slot_mask}'

                path = export_dir
                practical_path = path
                if not path.startswith("/"):
                    if args.context_dir.startswith("/"):
                        practical_path = os.path.abspath(path)
                    else:
                        practical_path = os.path.relpath(path, args.context_dir)

                obsfiledb.add_obsfile(
                    os.path.join(practical_path, os.path.split(filename)[1]),
                    obs_id,
                    detset,
                    0,
                    1,
                )
            # obs info
            obsdb.update_obs(obs_id, obs_info)
            print(f"  added {obs_id}")
            item_count += 1

            if args.test and item_count >= 5:
                break

        if args.test and item_count >= 5:
            break

    # detdb.to_file(f'{args.context_dir}/detdb.sqlite')
    obsdb.to_file(f"{args.context_dir}/obsdb.sqlite")
    obsfiledb.to_file(f"{args.context_dir}/obsfiledb.sqlite")

    #
    # metadata: det_info & focalplane
    #

    scheme = metadata.ManifestScheme()
    scheme.add_exact_match("dets:detset")
    scheme.add_data_field("dataset")
    db1 = metadata.ManifestDb(scheme=scheme)

    scheme = metadata.ManifestScheme()
    scheme.add_exact_match("dets:detset")
    scheme.add_data_field("dataset")
    db2 = metadata.ManifestDb(scheme=scheme)

    for detset, (props, fp) in detsets.items():
        key = "dets_" + detset
        props.keys = ["dets:" + k for k in props.keys]
        write_dataset(props, f"{args.context_dir}/metadata.h5", key, overwrite=True)
        db1.add_entry({"dets:detset": detset, "dataset": key}, filename="metadata.h5")

        key = "focalplane_" + detset
        write_dataset(fp, f"{args.context_dir}/metadata.h5", key, overwrite=True)
        db2.add_entry({"dets:detset": detset, "dataset": key}, filename="metadata.h5")

    db1.to_file(f"{args.context_dir}/det_info.sqlite")
    db2.to_file(f"{args.context_dir}/focalplane.sqlite")

    # And the context.yaml!
    context = {
        "tags": {"metadata_lib": "./"},
        "imports": ["sotodlib.io.metadata"],
        "obsfiledb": "{metadata_lib}/obsfiledb.sqlite",
        #'detdb': '{metadata_lib}/detdb.sqlite',
        "obsdb": "{metadata_lib}/obsdb.sqlite",
        "obs_loader_type": "toast3-hdf",
        "obs_colon_tags": ["wafer_slot", "band"],
        "metadata": [
            {"db": "{metadata_lib}/det_info.sqlite", "det_info": True},
            {"db": "{metadata_lib}/focalplane.sqlite", "name": "focal_plane"},
        ],
    }

    if args.dichroic_sub:
        context["obs_loader_type"] = "toast3-hdf-dichroic-hack"

    if args.absolute:
        context["tags"]["metadata_lib"] = args.context_dir

    open(f"{args.context_dir}/context.yaml", "w").write(
        yaml.dump(context, sort_keys=False)
    )
