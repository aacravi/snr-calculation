
import numpy as np
import h5py
import pandas as pd

def load_run(filepath):
    run = {"meta": {}, "data": {}}

    with h5py.File(filepath, "r") as f:

        run["meta"]["attrs"] = dict(f.attrs)
        run["meta"]["T_obs_yr"] = f.attrs["T_obs"] / (365 * 24 * 3600)
        run["meta"]["snr_threshold"] = f.attrs.get("snr_threshold", None)
        run["meta"]["n_total_sources"] = f.attrs.get("n_total_sources", None)
        run["meta"]["iterations"] = f.attrs.get("iterations", None)

        run["data"]["global_fr"] = f["global_fr"][:]
        resolved_sources = []
        table_rows = []

        grp = f["resolved_sources"]

        for key in grp.keys():  
            g = grp[key]
            src = {"id": g.attrs["id"],
                "f0": g.attrs["f0"],
                "fdot": g.attrs["fdot"],
                "Ampl": g.attrs["Ampl"],
                "ecliptic_lat": g.attrs.get("ecliptic_lat", np.nan),
                "ecliptic_lon": g.attrs.get("ecliptic_lon", np.nan),
                "lum_dist": g.attrs.get("lum_dist", np.nan),
                "A": g["A"][:],
                "E": g["E"][:],
                "fr": g["fr"][:],
            }

            snr = g.attrs["snr"].real

            resolved_sources.append({"source": src, "snr": snr})

            table_rows.append({
                "id": src["id"],
                "f0": src["f0"],
                "fdot": src["fdot"],
                "Ampl": src["Ampl"],
                "snr": snr,
                "ecliptic_lat": src["ecliptic_lat"],
                "ecliptic_lon": src["ecliptic_lon"],
                "lum_dist": src["lum_dist"]
            })

        run["data"]["resolved_sources"] = resolved_sources
        run["data"]["resolved_table"] = pd.DataFrame(table_rows)

        psd_iter = {}
        grp_psd = f["psd_confusion"]
        for key in grp_psd.keys():
            psd_iter[key] = grp_psd[key]["psd_total"][:]

        run["data"]["psd_iter"] = psd_iter

        hist_grp = f["history"]
        run["data"]["history"] = {name: hist_grp[name][:] for name in hist_grp.keys()}

    return run
