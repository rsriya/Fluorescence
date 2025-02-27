import cutax
import strax
st = cutax.xenonnt_online()
st.storage.append(strax.DataDirectory("/project2/lgrandi/xenonnt/processed/", readonly=True))
st.storage.append(strax.DataDirectory("/project/lgrandi/xenonnt/processed/", readonly=True))


from tqdm import trange
import numpy as np
import pandas as pd
import sys
import os

run = sys.argv[1]
#run = '053502'

import power_law_functions as plf
min_primary_size_PE = 100 #min size of primary peaks (PE)
overlap_cut_time_s = 3e-1 #the time constant for exponential overlap cut, https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:adepoian:overlapcut_overview
overlap_cut_ratio = 0.1 #peaks are only accepted if all previous primary peaks multiplied by (t/drift_time)^(overlap_cut_power) is less than overlap_cut_ratio of the current peak
overlap_cut_max_times = 30 #number of time constants to consider previous peaks for overlap cut
reject_edge_time_ns = 9 * 1e9

def lower_boundary(area):
    # Create an array to store the results
    cs2_area_fraction_top = np.zeros_like(area)

    # Apply conditions using boolean indexing
    mask1 = area < 2E4
    mask2 = (area >= 2E4) & (area < 1E5)
    mask3 = (area >= 1E5) & (area < 1E6)
    mask4 = area >= 1E6

    # Element-wise operations for each condition
    cs2_area_fraction_top[mask1] = 0.7645-1.914 / np.sqrt(area[mask1]) -1.52e-8*area[mask1]
    cs2_area_fraction_top[mask2] = 0.7607-1.3129 / np.sqrt(area[mask2]) +2.45e-9*area[mask2]
    cs2_area_fraction_top[mask3] = 0.75679
    cs2_area_fraction_top[mask4] = 0

    return cs2_area_fraction_top
    
def upper_boundary(area):
    # Create an array to store the results
    cs2_area_fraction_top = np.zeros_like(area)

    # Apply conditions using boolean indexing
    mask1 = area < 2E4
    mask2 = (area >= 2E4) & (area < 1E5)
    mask3 = (area >= 1E5) & (area < 1E6)
    mask4 = area >= 1E6
    
    # Element-wise operations for each condition
    cs2_area_fraction_top[mask1] = 0.7668+1.7915 / np.sqrt(area[mask1]) +5.76e-9*area[mask1]
    cs2_area_fraction_top[mask2] = 0.7692+1.1637 / np.sqrt(area[mask2]) +1.02e-9*area[mask2]
    cs2_area_fraction_top[mask3] = 0.77298
    cs2_area_fraction_top[mask4] = 0.77298

    return cs2_area_fraction_top

file_path = f'/scratch/midway3/astroriya/primaries/{run}.pkl'

if os.path.exists(file_path):
    #print("File exists.")
    selected_peaks = pd.read_pickle(f'/scratch/midway3/astroriya/primaries/{run}.pkl')
else:
    print(f"{run} does not exist.")
    data_peaks = st.get_df(run, ('peak_basics', 'peak_positions_mlp'))
    start_time = data_peaks['time'].min()
    end_time = data_peaks['time'].max()
    primaries = data_peaks.query('area > @min_primary_size_PE') 
    
    del data_peaks
    
    #overlap cut
    rejected_peak_times = plf.peaks_to_reject(primaries['area'].values, primaries['time'].values, overlap_cut_time_s=overlap_cut_time_s, overlap_cut_max_times=overlap_cut_max_times, overlap_cut_ratio=overlap_cut_ratio)

    mask = (
        ~primaries['time'].isin(rejected_peak_times) &  # Not in rejected times
        (primaries['time'] > start_time + reject_edge_time_ns) &  # Within valid start bound
        (primaries['time'] < end_time - reject_edge_time_ns) &    # Within valid end bound
        (primaries['type'] == 2)  # Type filtering
    )
    selected_peaks = primaries.loc[mask]  # Faster filtering             
    selected_peaks = selected_peaks[selected_peaks['area_fraction_top']<upper_boundary(selected_peaks['area'])]
    selected_peaks = selected_peaks[selected_peaks['area_fraction_top']>lower_boundary(selected_peaks['area'])]
    
    selected_peaks['window_lengths'] = plf.get_windows(primaries, selected_peaks, overlap_cut_time_s=overlap_cut_time_s, overlap_cut_max_times=overlap_cut_max_times, overlap_cut_ratio=overlap_cut_ratio)
    selected_peaks.to_pickle(f'/scratch/midway3/astroriya/primaries/{run}.pkl')
