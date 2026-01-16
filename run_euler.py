
import argparse
import gc
from pathlib import Path

import numpy as np
import h5py
import astropy.units as u

from library.snr import optimal_snr
from library.lisa_psd import noise_psd_AE
from library.iteration_utils import (
    load_waveforms,
    setup,
    run_iterative_separation,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run iterative separation on LISA data")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input HDF5 file with processed waveforms",
    )

    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=7.0,
        help="SNR threshold for source selection",
    )

    parser.add_argument(
        "--filter-size",
        type=int,
        default=2000,
        help="Filter size for the separation algorithm",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum number of iterations",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Base name for output results (no extension)",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable plotting (NOT recommended on Euler)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_file = Path(args.input)
    output_file = args.output
    snr_thr = args.snr_threshold
    filter_size = args.filter_size
    max_iterations = args.max_iterations

    print("Running with parameters:")
    print(f"  input          = {input_file}")
    print(f"  snr_threshold  = {snr_thr}")
    print(f"  filter_size    = {filter_size}")
    print(f"  max_iterations= {max_iterations}")
    print(f"  output         = {output_file}")
    print(f"  plot           = {args.plot}")

    # --- Load data
    data = load_waveforms(input_file)
    T_obs = data["T_obs"]

    # --- Setup state
    state = setup(
        data,
        snr_calculator=lambda source: optimal_snr(
            source["A"], source["psd_total"], T_obs=T_obs
        ),
        psd_instrumental=noise_psd_AE,
        snr_threshold=snr_thr,
        filter_size=filter_size,
    )

    # Free memory early (important on Euler)
    del data
    gc.collect()

    # --- Run algorithm
    results = run_iterative_separation(
        state,
        max_iterations=max_iterations,
        filter_size=filter_size,
        print_progress=True,
        plot=args.plot,          # default OFF
        save_results=True,
        output_file=output_file,
    )

    print("Run finished successfully.")


if __name__ == "__main__":
    main()
