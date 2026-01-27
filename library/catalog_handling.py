import numpy as np
import h5py
from library.snr import approx_snr
from library.lisa_psd import psd_source_approx
from library.lisa_response_fastGB import  tdi_AE_fastGB_multi
from scipy.constants import c
from tqdm import tqdm
import gc


def characteristic_strain(T_obs, f0, Amp):
    h_c = (16 /5 * T_obs)**(1/2) * f0 * Amp
    return h_c


def process_catalog_batches(catalog, T_obs, delta_t, tdi,  batch_size, output_file = 'galactic_binaries_waveforms.hdf5', snr_preselection = 0.001, verbose=True):
    """
    Function to process the catalog in batches

    Parameters:
    ------------------
    catalog: contains binaries parameters
    T_obs: in s
    delta_t: sampling time in s
    batch_size: 
    output_file: name of output file
    snr_preselection: threshold for pre-exclusion of sources to avoid calculating waveform
    ------------------
    Returns: output file with waveforms, psd estimate, position of source, source's properties, observational parameters
    """

    n_tot = len(catalog["Amplitude"])
    n_batches = int(np.ceil(n_tot/batch_size))

    if verbose:
        print(f"Processing {n_tot} sources")
        print(f"Batches: {n_batches} batches with {batch_size} sources")
    
    # Create different N values to assign depending on the required N (depends on the fdot)
    N_values = {
    "small": 64,
    "medium": 256,
    "large": 4096
    }
    bucket_items = list(N_values.items())

    with h5py.File(output_file, 'w') as f:

        # This creates a dataset for all sources, where I save the waveforms and useful parameters of each batch. Save them in buckets of different Ns to save space
        # do NOT store individual PSDs 
        meta = f.create_group('meta')
        meta_f0 = meta.create_dataset('f0', shape=(n_tot,), dtype='float64')
        meta_fdot = meta.create_dataset('fdot', shape=(n_tot,), dtype='float64')
        meta_ampl = meta.create_dataset('Ampl', shape=(n_tot,), dtype='float64')
        

        psd_est = f.create_dataset('source_psd_estimate', shape=(n_tot,), dtype='float64')

        for bname, N in bucket_items:
            grp = f.create_group(bname)
            grp.create_dataset('indices', shape=(0,), maxshape=(None,), dtype='int') # to keep track of the source in the whole catalog
            grp.create_dataset('A', shape=(0, N), maxshape=(None, N), dtype='complex128')
            grp.create_dataset('E', shape=(0, N), maxshape=(None, N), dtype='complex128')
            grp.create_dataset('fr', shape=(0, N), maxshape=(None, N), dtype='float64')
            grp.create_dataset('ecliptic_lat', shape=(0,), maxshape=(None,), dtype='float64')
            grp.create_dataset('ecliptic_lon', shape=(0,), maxshape=(None,), dtype='float64')
    
        # Store observational parameters as attributes
        f.attrs['T_obs'] = T_obs
        f.attrs['N_values'] = list(N_values.values())
        f.attrs['delta_t'] = delta_t

        # Process the batch
        for i in tqdm(range(n_batches), desc="Processing batches", disable=not verbose, mininterval=1.0):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_tot)
            current_batch_size = end_idx - start_idx
            
            # Extract this batch from the full catalog
            # Assign random phases phi0
            phi0 = np.random.uniform(0, 2*np.pi, current_batch_size)

            batch_params = np.column_stack([
                catalog['GW22FrequencySourceFrame'][start_idx:end_idx],
                catalog['GW22FrequencyDerivativeSourceFrame'][start_idx:end_idx],
                catalog['Amplitude'][start_idx:end_idx],
                catalog['EclipticLatitude'][start_idx:end_idx],
                catalog['EclipticLongitude'][start_idx:end_idx],
                catalog['PolarisationAngle'][start_idx:end_idx],
                catalog['InclinationAngle'][start_idx:end_idx],
                phi0
            ])
            
            meta_f0[start_idx:end_idx] = batch_params[:, 0]
            meta_fdot[start_idx:end_idx] = batch_params[:, 1]
            meta_ampl[start_idx:end_idx] = batch_params[:, 2]
            
            # Rough estimation of the SNR of sources to avoid computing the waveform of already weak sources
            # For weak sources, compute the estimated PSD and store it (it will contribute to the background)
            fdot = batch_params[:, 1]
            f0 = batch_params[:,0]
            amp = batch_params[:,2]

            h_c = characteristic_strain(T_obs, f0, amp)
            SNR_approx = approx_snr(h_c, f0)
            psd_est[start_idx:end_idx] = psd_source_approx(h_c, f0, tdi)

            # Mask the loud sources (possibly resolvable) using a SNR threshold. Skipped sources have None instead of the waveform
            loud_sources_mask = SNR_approx > snr_preselection
                
            # Assign N based on fdot to compute the waveform with FastGB for the loud sources
            required_bins = np.abs(fdot) * T_obs**2  

            bucket = np.full(len(f0),'skip', dtype='U10') # initialize the bukcets with all skip

            bucket[(required_bins < 1e3) & loud_sources_mask] = "small"
            bucket[(required_bins >= 1e3) & (required_bins < 1e4) & loud_sources_mask] = "medium"
            bucket[(required_bins >= 1e4) & loud_sources_mask] = "large"

            # process each bucket individually
            for bname, N in N_values.items():
                mask = (bucket == bname)
                if not np.any(mask):
                    continue

                params_sub = batch_params[mask]
                idxs = mask.nonzero()[0]
                global_idxs = start_idx + idxs

                A_sub, E_sub, kmin, fr_sub = tdi_AE_fastGB_multi(
                    params_sub,
                    delta_t=delta_t,
                    T_obs=T_obs,
                    N=N,
                    tdi=tdi
                )
                # Save position for sources with waveform
                ecliptic_lat_sub = params_sub[:,3] 
                ecliptic_lon_sub = params_sub[:,4]

                grp = f[bname]
                old = grp['indices'].shape[0]
                new = old + len(global_idxs)
                
                for dset in ('indices', 'ecliptic_lat', 'ecliptic_lon'):
                    grp[dset].resize((new,))
                for dset in ('A', 'E', 'fr'):
                    grp[dset].resize((new, N))
            

                grp['indices'][old:new] = global_idxs
                grp['A'][old:new] = A_sub
                grp['E'][old:new] = E_sub
                grp['fr'][old:new] = fr_sub
                grp['ecliptic_lat'][old:new] = ecliptic_lat_sub  
                grp['ecliptic_lon'][old:new] = ecliptic_lon_sub

                # Clear memory
                del A_sub, E_sub, fr_sub, ecliptic_lat_sub, ecliptic_lon_sub
            if i % 1000 == 0: 
                gc.collect()
        if verbose:
            print(f"Saved {n_tot} waveforms sources to {output_file}")
            print(f"File size: {f.id.get_filesize() / (1024**3):.2f} GB")
        
    return output_file