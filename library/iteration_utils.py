import numpy as np
import matplotlib.pyplot as plt
from library.lisa_psd import noise_psd_AE, noise_psd_AE_gal2
import scipy.constants as constants
from library.snr import optimal_snr
import h5py
from scipy import ndimage
from pathlib import Path


def load_waveforms(filename, distance_cut=None):

    with h5py.File(filename, 'r') as f:
        n_tot = len(f['meta/f0'])

        dist = f['meta/lum_dist'][:] * 1000

        if distance_cut is None:
            keep = np.ones(n_tot, dtype=bool)
        else:
            keep = dist <= distance_cut

        ids_keep = np.where(keep)[0]
        n_keep = len(ids_keep)

        global_id = ids_keep.copy() # keep original indices to identify sources in the full catalog

        # Pre-allocate lists
        A = [None] * n_keep
        E = [None] * n_keep
        fr = [None] * n_keep
        ecl_lat = [None] * n_keep  
        ecl_lon = [None] * n_keep
        with_wf = np.zeros(n_keep, dtype = bool) # Track only sources with a waveform 

        old_to_new = -np.ones(n_tot, dtype=int)
        old_to_new[ids_keep] = np.arange(n_keep)

        for bucket in ['small', 'medium', 'large']:
            grp = f[bucket]
            idx = grp['indices'][:]

            A_vals  = grp['A'][:]
            E_vals  = grp['E'][:]
            fr_vals = grp['fr'][:]
            lat_vals = grp['ecliptic_lat'][:]
            lon_vals = grp['ecliptic_lon'][:]

            for j, gidx in enumerate(idx):

                new_idx=old_to_new[gidx]
                if new_idx == -1:
                    continue

                A[new_idx] = A_vals[j]
                E[new_idx] = E_vals[j]
                fr[new_idx] = fr_vals[j]
                ecl_lat[new_idx] = lat_vals[j]
                ecl_lon[new_idx] = lon_vals[j]
                with_wf[new_idx] = True

        data = {
            'A': A,
            'E': E,
            'fr': fr,
            'source_psd_estimate': f['source_psd_estimate'][:][keep],
            'f0': f['meta/f0'][:][keep],
            'Ampl': f['meta/Ampl'][:][keep],
            'fdot': f['meta/fdot'][:][keep],
            'with_wf': with_wf,
            'ecliptic_lat': ecl_lat,
            'ecliptic_lon': ecl_lon,
            'lum_dist': dist[keep],
            'id': global_id,
            'T_obs': f.attrs['T_obs']
        }
                
    return data



def power_spectrum_calc(h, f):
    """
    Function to compute the power spectrum of data

    Parameters
    ----------
    h : array_like
        Strain data
        
    f : array_like
         frequency of data

    Returns
    -------
    psd : One-sided PSD [strain^2 / Hz]
    """
    h = np.asarray(h)
    f = np.asarray(f)
    
    # Frequency bin width
    df = f[1] - f[0]

    psd = 2* np.abs(h)**2 *df

    return psd

from scipy.ndimage import gaussian_filter1d


def pre_excluded_conf(sources, weak_indices, global_fr, filter_size=1000):
    """ 
    To calculate the confusion PSD of the pre-excluded sources (from estimated PSD)
    """
    psd_conf = np.zeros_like(global_fr)

    if len(weak_indices) == 0:
        return psd_conf
    
    psd_estimates = sources['source_psd_estimate'][weak_indices]
    f0_values = sources['f0'][weak_indices]

    freq_indices = np.searchsorted(global_fr, f0_values)
    freq_indices = np.clip(freq_indices, 0, len(global_fr) -1 )

    np.add.at(psd_conf, freq_indices, psd_estimates)

    psd_conf_smooth = gaussian_filter1d(psd_conf, sigma=filter_size/2.355/2)
    
    return psd_conf_smooth


def setup(sources, snr_calculator, psd_instrumental, snr_threshold=7.0, tdi=1.5,  filter_size=1000):
    """
    Setup for the iteration
    
    Parameters:
    -----------
    sources : dictionary with sources parameters              
    snr_calculator : function
    psd_instrumental : function
    df: delta_frequency for the global grid
    snr_threshold : float
        SNR threshold (default is 7.0)
        
    Returns:
    --------
    state : dictionary
        Contains all data needed for iteration
    """
    
    n_sources = len(sources['f0'])
    with_wf = sources.get('with_wf', np.ones(n_sources, dtype=bool))
    
    # Separate indices of sources with/without waveform
    wf_indices = np.where(with_wf)[0]
    weak_indices = np.where(~with_wf)[0]

    # Create global frequency grid (useful for global PSD calculation)
    df = None
    freq_min = np.inf
    freq_max = -np.inf

    for idx in wf_indices:
        fr = sources['fr'][idx]
        if df is None:
            df = fr[1] - fr[0]
        freq_min = min(freq_min, fr[0])
        freq_max = max(freq_max, fr[-1])

    global_fr = np.arange(freq_min - df, freq_max + 2*df, df)    

    source_A = {}
    source_E = {}
    source_idx_ranges = {}
    for idx in wf_indices:
        fr = sources['fr'][idx]
        A  = sources['A'][idx]   # complex FD waveform
        E = sources['E'][idx]

        idx_start = np.searchsorted(global_fr, fr[0])
        idx_end   = idx_start + len(fr)

        source_A[idx] = A
        source_E[idx] = E
        source_idx_ranges[idx] = (idx_start, idx_end)
        
    # Calculate global PSD instr and confusion from pre-excluded sources
    psd_instr_global = psd_instrumental(global_fr, tdi=tdi)
    psd_conf_preexcluded = pre_excluded_conf(sources, weak_indices, global_fr, filter_size=1000)
    
    state = {
        'waveforms': sources,
        'source_A': source_A,
        'source_E':source_E,
        'source_idx_ranges': source_idx_ranges,
        'idx_unresolved': wf_indices.copy(),
        'idx_pre_excluded': weak_indices,
        'sources_resolved': [],
        'idx_confusion_only': [],
        'idx_snr_candidates' : wf_indices.copy(),
        'psd_instr_global': psd_instr_global,
        'psd_conf_preexcluded': psd_conf_preexcluded,
        'calculate_snr': snr_calculator,
        'snr_threshold': snr_threshold,
        'iteration': 0,
        'history': [],
        'global_fr': global_fr,
        'df': df,
        'freq_min': global_fr.min(),
        'freq_max': global_fr.max()
    }
    
    return state


def confusion_psd_from_signal(A_tot, df,  filter_size=1000):
    """
    Compute confusion PSD from total signal

    Parameters:

    A_tot: total signal of  channel A=channel E 
    df: frequency resolution (1/T_obs)
    filter_size: number of bins for the smoothing 

    """
    signal_psd = 2.0 * df * np.abs(A_tot**2)
    #norm = 1.0 / 0.7023319615912207
    psd_conf = ndimage.median_filter(signal_psd, size=filter_size) 

    return psd_conf


def separate_snr(sources, indices, psd_total_global,  global_fr, calculate_snr_function, threshold):
    """
    Separate sources into resolved vs unresolved based on SNR
        
    Returns:
    --------
    resolved : Resolved sources with their SNRs
    unresolved_idx : indices of unresolved sources to use in the next iteration
    """
    
    resolved = []
    unresolved_idx = []
    
    for idx in indices:
        fr = sources['fr'][idx]

        idx_start = np.searchsorted(global_fr, fr[0])
        idx_end   = idx_start + len(fr)

        psd_source = psd_total_global[idx_start:idx_end]
        source = {
            'f0': sources['f0'][idx],
            'fdot': sources['fdot'][idx],
            'Ampl': sources['Ampl'][idx],
            'fr': fr,
            'A': sources['A'][idx],
            'E':sources['E'][idx],
            'psd_total': psd_source,
            'id': sources['id'][idx],
            'local_idx': idx,
            'ecliptic_lon':sources['ecliptic_lon'][idx],
            'ecliptic_lat':sources['ecliptic_lat'][idx],
        }

        snr = calculate_snr_function(source)
        
        if snr >= threshold:
            resolved.append({'source': source, 'snr': snr})
        else:
            unresolved_idx.append(idx)
    
    return resolved, np.array(unresolved_idx)

def save_results_h5(output_file, results, state):
    with h5py.File(output_file, "w") as f:

        f.attrs["T_obs"] = state.get("T_obs", state["waveforms"].get("T_obs", 0))
        f.attrs["snr_threshold"] = state["snr_threshold"]
        f.attrs["n_total_sources"] = len(state["waveforms"]["f0"])
        f.attrs["iterations"] = results["iterations"]

        f.create_dataset("global_fr", data=results["global_fr"])

        grp = f.create_group("resolved_sources")

        for i, r in enumerate(results["resolved_sources"]):
            src = r["source"]
            g = grp.create_group(f"source_{i}")

            g.attrs["id"] = src["id"]
            g.attrs["f0"] = src["f0"]
            g.attrs["fdot"] = src["fdot"]
            g.attrs["Ampl"] = src["Ampl"]
            g.attrs["snr"] = r["snr"]
            g.attrs["ecliptic_lat"] = src.get("ecliptic_lat", np.nan)
            g.attrs["ecliptic_lon"] = src.get("ecliptic_lon", np.nan)

            g.create_dataset("A", data=src["A"])
            g.create_dataset("E", data=src["E"])
            g.create_dataset("fr", data=src["fr"])
            
        grp_psd = f.create_group("psd_confusion")

        for iteration, psd in results["psd_confusion"]:
            g = grp_psd.create_group(f"iter_{iteration}")
            g.create_dataset("psd_total", data=psd)

        f.create_dataset("unresolved_indices", data=np.array(results["unresolved_indices"], dtype=int))

        hist = results["history"]
        grp_hist = f.create_group("history")
        grp_hist.create_dataset("iteration", data=[h["iteration"] for h in hist])
        grp_hist.create_dataset( "n_resolved_this_step",  data=[h["n_resolved_this_step"] for h in hist])
        grp_hist.create_dataset("n_unresolved_remaining", data=[h["n_unresolved_remaining"] for h in hist])
        grp_hist.create_dataset( "n_resolved_total",data=[h["n_resolved_total"] for h in hist])

    print(f"Results saved to {output_file}")


def run_iterative_separation(state,  
                             max_iterations=100, 
                             filter_size=1000,
                             print_progress=True,
                             plot = True,
                             save_results= True,
                             output_file= 'resolved_sources.hdf5'):
    """    
    Parameters:
    -----------
    state : dictionary
        State from setup_separation()
    max_iterations : int
    confusion_method : str
        'median' or 'mean'
    print_progress : bool
        Print what's happening at each step
        
    Returns:
    --------
    results : dictionary
        Final results with resolved/unresolved sources
    Saves file with resolved sources
    """

    n_tot_sources = len(state['waveforms']['f0'])
    n_pre_confusion = len(state['idx_pre_excluded'])

    if print_progress:
        print(f"Starting with {n_tot_sources} total sources")
        print(f"{n_pre_confusion} sources already pre-excluded and added to the confusion")
        print(f"{len(state["idx_unresolved"])} candidate sources to run the iteration")
        print(f"Frequency range: {state['freq_min']:.2e} to {state['freq_max']:.2e} Hz")
        print(f"SNR threshold: {state['snr_threshold']}")
           
    n_resolved_previous = len(state['sources_resolved'])
    global_fr = state['global_fr']
    df = global_fr[1] - global_fr[0]
    psd_confusion_iter = [] # store total PSD (with confusion) for each iteration

    # STEP 0: Instrument-only SNR to exclude unresolvable sources
    snr_candidates, unresolved_instr = separate_snr(
        state['waveforms'],
        state['idx_unresolved'],
        state['psd_instr_global'],   
        state['global_fr'],
        state['calculate_snr'],
        state['snr_threshold']/2
    )

    state['idx_unresolved'] = [r['source']['local_idx'] for r in snr_candidates]
    state['idx_confusion_only'] = list(unresolved_instr)
    state['idx_unresolved'] = list(state['idx_unresolved'])
    state['idx_confusion_only'] = list(state['idx_confusion_only'])

    if print_progress:
        print("\n--- Step 0: Instrument-only SNR pre-exclusion")
        print(f"  Unresolvable sources (added to confusion): {len(state['idx_confusion_only'])}")
        print(f"  SNR candidates for iteration: {len(snr_candidates)}")

    # MAIN ITERATION LOOP
    for iteration in range(1, max_iterations + 1):
        
        if print_progress:
            print(f"\n--- Iteration {iteration} ---")
        
        # STEP 1: Calculate global confusion PSD (A=E)
        A_tot = np.zeros_like(state['global_fr'], dtype=np.complex128)

        confusion_idx = (
            list(state['idx_unresolved']) +
            list(state['idx_confusion_only'])
            )

        for idx in confusion_idx:
            A = state['source_A'][idx]
            i0, i1 = state['source_idx_ranges'][idx]
            A_tot[i0:i1] += A


        psd_confusion= confusion_psd_from_signal(
            A_tot=A_tot,
            df=state['df'],
            filter_size=filter_size
        )

        psd_instr=state['psd_instr_global']

        #psd_confusion_global = psd_confusion
        psd_confusion_global = (
            psd_confusion +
            state['psd_conf_preexcluded']
        )
        
        psd_total_global = psd_confusion_global + psd_instr
        psd_confusion_iter.append((iteration, psd_total_global))

        if plot:
            print(f"Size of the binning for the median filter: {filter_size*df}")
            plt.loglog(global_fr, psd_instr, color = 'black', label="Instrumental PSD")
            #plt.loglog(global_fr, state['psd_conf_preexcluded'], color = 'orange', alpha = 0.7, label ='confusion pre-excluded')
            plt.loglog(global_fr, psd_confusion_global, color = 'red', alpha = 0.5, label='Confusion unresolved')
            plt.loglog(global_fr, psd_total_global, color ='teal', alpha = 0.7 ,label="Confusion + instr PSD")
            #plt.ylim(1e-48, 1e-37)
            plt.title(f"Iteration {iteration}, filter size {filter_size}, bin size {(filter_size*df):.2e}")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.show()
        
        # STEP 2: Separate sources with updated confusion
        resolved_new, unresolved_idx = separate_snr(
            state['waveforms'],
            state['idx_unresolved'],
            psd_total_global,
            global_fr,
            state['calculate_snr'],
            state['snr_threshold']
        )
        
        # STEP 3: Update state 
        state['sources_resolved'].extend(resolved_new)
        state['idx_unresolved'] = unresolved_idx
        
        n_resolved_now = len(state['sources_resolved'])
        
        state['history'].append({
            'iteration': iteration,
            'n_resolved_this_step': len(resolved_new),
            'n_unresolved_remaining': len(state['idx_unresolved']),
            'n_resolved_total': n_resolved_now
        })
        
        if print_progress:
            print(f"  New resolved: {len(resolved_new)}")
            print(f"  Still unresolved: {len(state['idx_unresolved'])}")
            print(f"  Total resolved: {n_resolved_now}")
        
        # STEP 4: Check convergence
        n_newly_resolved = n_resolved_now - n_resolved_previous
        
        if n_newly_resolved == 0:
            if print_progress:
                print(f"\nConverged after {iteration} iterations!")
            break
        
        n_resolved_previous = n_resolved_now
    
    else:
        if print_progress:
            print(f"\nReached maximum iterations ({max_iterations})")
    
    resolved_global_ids = [r['source']['id'] for r in state['sources_resolved']]

    results = {
        'resolved_sources': state['sources_resolved'],
        'global_fr': global_fr,
        'psd_confusion': psd_confusion_iter,
        'resolved_global_indices': resolved_global_ids,
        'n_resolved': len(state['sources_resolved']),
        'n_unresolved': len(state['idx_unresolved']),
        'iterations': iteration,
        'history': state['history']
    }
    
    if print_progress:
        print("\n" + "=" * 60)
        print("FINAL RESULTS:")
        print(f"  Total sources: {n_tot_sources}")
        print(f"  Resolved: {results['n_resolved']} ({results['n_resolved']/n_tot_sources*100:.1f}%)")
        print(f"  Unresolved: {n_tot_sources-results['n_resolved']} ({100 - results['n_resolved']/n_tot_sources*100:.1f}%)")
        print("=" * 60)
    

    if save_results:
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        output_path = results_dir / output_file
        if print_progress:
            print(f"\nSaving resolved sources to results/{output_file}...")
        
        save_results_h5(output_path, results, state)

    return results