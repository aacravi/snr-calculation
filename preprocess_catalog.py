# preprocess_catalog.py
import h5py
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
from library.catalog_handling import process_catalog_batches
import gc
import argparse

def preprocess_catalog(filepath, output_filename,
                       keys=None, T_obs=4*365*24*3600,
                       delta_t=5, tdi=1.5, snr_preselection=0.01,
                       batch_size=1000):
    """
    Load the catalog, convert coordinates, and process waveforms.
    Saves the waveforms to `output_filename`.
    """
    print("Loading catalog...")
    if keys is None:
        keys = ["GW22FrequencySourceFrame", "GW22FrequencyDerivativeSourceFrame",
                "Amplitude", "Declination", "RightAscension", "PolarisationAngle",
                "InclinationAngle", "SecondaryMassSSBFrame", "PrimaryMassSSBFrame",
                "TotalMassSSBFrame"]

    # Load catalog
    with h5py.File(filepath, "r") as f:
        binaries = f["Binaries"]
        param_binaries = {name: binaries[name][:] for name in keys}

    # Coordinate conversion
    ra = param_binaries.pop("RightAscension")
    dec = param_binaries.pop("Declination")
    c = SkyCoord(ra=ra, dec=dec, unit='rad', frame='icrs')
    ecl = c.transform_to(GeocentricTrueEcliptic())
    param_binaries["EclipticLongitude"] = ecl.lon.rad
    param_binaries["EclipticLatitude"] = ecl.lat.rad

    # Clean memory
    del c, ecl, ra, dec
    gc.collect()

    print("Processing catalog...")
    # Process catalog in batches batches
    process_catalog_batches(param_binaries,
                            T_obs=T_obs,
                            delta_t=delta_t,
                            tdi=tdi,
                            batch_size=batch_size,
                            snr_preselection=snr_preselection,
                            output_file=output_filename,
                            verbose=True)

    # Clean memory
    del param_binaries
    gc.collect()
    print(f"Catalog pre-processing done. Output saved to {output_filename}")


# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess LISA binary catalog")

    # Required
    parser.add_argument("--filepath", required=True, help="Path to HDF5 catalog")
    parser.add_argument("--output", required=True, help="Output filename")

    # Optional parameters with defaults
    parser.add_argument("--T_obs", type=float, default=4*365*24*3600, help="Observation time in seconds")
    parser.add_argument("--delta_t", type=float, default=5, help="Time resolution in seconds")
    parser.add_argument("--tdi", type=float, default=1.5, help="TDI scaling factor")
    parser.add_argument("--snr_preselection", type=float, default=0.01, help="SNR preselection threshold")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--keys", nargs="+", default=None, help="List of catalog keys to load (space-separated). If not provided, default keys are used.")

    args = parser.parse_args()
    
    preprocess_catalog(args.filepath,
                       args.output,
                       keys=args.keys,
                       T_obs=args.T_obs,
                       delta_t=args.delta_t,
                       tdi=args.tdi,
                       snr_preselection=args.snr_preselection,
                       batch_size=args.batch_size)  