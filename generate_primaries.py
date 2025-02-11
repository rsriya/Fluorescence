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
max_drift_time_s = 2.5e-3 #s
SE_gain = 31 #https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:abby:segain_evolution

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
    print("File exists.")
    selected_peaks = pd.read_pickle(f'/scratch/midway3/astroriya/primaries/{run}.pkl')
else:
    print("File does not exist.")
    data_peaks = st.get_df(run, ('peak_basics', 'peak_positions_mlp'))
    primaries = data_peaks.query('area > @min_primary_size_PE') 
    
    del data_peaks
    
    #overlap cut
    rejected_peak_times = plf.peaks_to_reject(primaries['area'].values, primaries['time'].values, overlap_cut_time_s=overlap_cut_time_s, overlap_cut_max_times=overlap_cut_max_times, overlap_cut_ratio=overlap_cut_ratio)
    
    selected_peaks = primaries.query('time not in @rejected_peak_times')
     
    
    selected_peaks = selected_peaks[selected_peaks['type']==2]
    
    
    
    selected_peaks = selected_peaks[selected_peaks['area_fraction_top']<upper_boundary(selected_peaks['area'])]
    selected_peaks = selected_peaks[selected_peaks['area_fraction_top']>lower_boundary(selected_peaks['area'])]
    
    selected_peaks['window_lengths'] = plf.get_windows(primaries, selected_peaks, overlap_cut_time_s=overlap_cut_time_s, overlap_cut_max_times=overlap_cut_max_times, overlap_cut_ratio=overlap_cut_ratio)
    del primaries
    selected_peaks.to_pickle(f'/scratch/midway3/astroriya/primaries/{run}.pkl')

data_peaks = st.get_df(run, ('peak_basics', 'peak_positions_mlp'))
S2_peaks = data_peaks[data_peaks['type']==2]
del data_peaks
S2_peaks = S2_peaks[S2_peaks['tight_coincidence']>=2]
ionization_rates = plf.measure_photoionization_rate_within_drift_time(SE_gain, max_drift_time_s, selected_peaks.query('window_lengths > @max_drift_time_s'), S2_peaks)
ionization_rate_mean = np.mean(ionization_rates)
s2_area_cut = [15,53]
aft_cut = [0.40, 0.99]
width_cut = [90, 650]
SE_data = S2_peaks.query('area < @s2_area_cut[1] and area > @s2_area_cut[0] and area_fraction_top < @aft_cut[1] and area_fraction_top > @aft_cut[0] and range_50p_area < @width_cut[1] and range_50p_area > @width_cut[0]')
del S2_peaks

def generate_powerlaw_dt(min_dt, max_dt, alpha, size):
    """
    Generates time differences (dt) following a power law distribution ~ dt^(-alpha).
    
    Args:
        min_dt (float): The minimum value for dt (to avoid infinity).
        max_dt (float): The maximum value for dt.
        alpha (float): The power law exponent.
        size (int): Number of samples to generate.
        
    Returns:
        np.array: Generated time differences (dt).
    """
    # Generate uniform random numbers between 0 and 1
    u = np.random.uniform(0, 1, size)

    # Power law sampling for dt ~ dt^(-alpha)
    dt = (min_dt**(1 - alpha) + u * (max_dt**(1 - alpha) - min_dt**(1 - alpha)))**(1 / (1 - alpha))

    return dt

def generate_signal_events(selected_peak, num_events, alpha, min_dt=5e6, max_dt=4e8):
    """
    Generates signal events based on the power law rate A * dt**(-alpha).
    
    Args:
        selected_peak (pd.Series): The row from selected_peaks representing the chosen event.
        num_events (int): Number of signal events to generate.
        A (float): The scaling factor for event rate.
        alpha (float): The exponent for the power law distribution.
        min_dt (float): The minimum time difference (to avoid division by zero).
        max_dt (float): The maximum time difference.
        
    Returns:
        pd.DataFrame: Generated signal events with 'x', 'y', and 'time'.
    """
    # Create an empty DataFrame for signal events
    signal_events = pd.DataFrame()

    # Generate power-law-distributed time differences
    dt = generate_powerlaw_dt(min_dt, max_dt, alpha, num_events)
    times = selected_peak['time'] + dt

    # Generate random x and y values for the signal events
    x_values = np.random.uniform(min(SE_data['x']), max(SE_data['x']), size=num_events)
    y_values = np.random.uniform(min(SE_data['y']), max(SE_data['y']), size=num_events)

    signal_events['x'] = x_values
    signal_events['y'] = y_values
    signal_events['time'] = times
    signal_events['sprinkled'] = True

    return signal_events

num_events = 1#round(np.sum((np.array(SE_dts)>5e6)&(np.array(SE_dts)<4e8))/(2*len(selected_peaks)))

all_signal_events = pd.DataFrame(columns=['x', 'y', 'time'])

# Loop through each row in selected_peaks
for _, selected_peak in selected_peaks.iterrows():
    # Generate signal events for the current selected peak
    signal_events = generate_signal_events(selected_peak, num_events, alpha=0.7)
    all_signal_events = pd.concat([all_signal_events, signal_events], ignore_index=True)

# Combine with data_SE
combined_data = pd.concat([SE_data, all_signal_events], ignore_index=True)

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

data_for_clustering = combined_data[['x', 'y','time']].values
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_for_clustering)
eps = 0.02
clustering = DBSCAN(eps=eps, min_samples=3, metric='euclidean').fit(data_normalized)
combined_data['cluster'] = clustering.labels_
combined_data = combined_data[combined_data['cluster'] == -1]
remaining_signal_events = combined_data[combined_data['sprinkled'].notna()]  # Only the introduced signal events have 'rate' values
acceptance = len(remaining_signal_events)/len(all_signal_events)
SE_data = combined_data[combined_data['sprinkled'].isna()]
SE_data = SE_data.drop(columns=['cluster'])

SE_times, SE_dts, SE_primary_areas, SE_primary_times = plf.get_lone_hit_times(selected_peaks, selected_peaks['window_lengths'].values , SE_data)

sorted_windows = np.sort(selected_peaks['window_lengths'].values)
window_bins_edges = np.concatenate([[0], sorted_windows])
number_of_overlapping_bins = np.arange(len(window_bins_edges)-1, 0, -1)
weights = 1/number_of_overlapping_bins
histogram_bins = np.logspace(-6, 0.5, 50)

SE_weights = weights[np.searchsorted(window_bins_edges, np.array(SE_dts)*1e-9, side='right')-1]/np.array(SE_primary_areas)
weighted_hist_SE, hist_errs_SE = plf.histogram_with_weights(np.array(SE_dts), np.array(SE_weights), histogram_bins)
start_bin = 28
end_bin = 43
SE_rate = np.sum(weighted_hist_SE[start_bin:end_bin]*np.diff(histogram_bins)[start_bin:end_bin])
SE_rate_var = (np.sum((hist_errs_SE[start_bin:end_bin]*np.diff(histogram_bins)[start_bin:end_bin])**2))


matched_SE = SE_data.query('time in @SE_times')
position_difference = np.zeros((len(SE_times), 2))
primary_positions = np.zeros((len(SE_times), 2))
for i in trange(len(SE_times)):
    peak_time = SE_times[i]
    primary_time = SE_primary_times[i]
    peak_position = matched_SE.iloc[i][['x', 'y']].values
    primary_position = selected_peaks.query('time == @primary_time').iloc[0][['x', 'y']].values
    position_difference[i] = primary_position - peak_position
    primary_positions[i] = primary_position
ds = np.sqrt(np.sum(position_difference**2, axis=1))
del selected_peaks

def get_area_normalization(radial_positions, radius, cut_radius):
    output = np.zeros_like(radial_positions)
    bool_tpc_radius_intersect = radial_positions + cut_radius > radius
    output += np.logical_not(bool_tpc_radius_intersect).astype(int)*(1 - cut_radius**2/radius**2)
    theta = 2*np.arccos((radius**2 - cut_radius**2 + radial_positions**2)/(2*radial_positions*radius), where=bool_tpc_radius_intersect)
    theta_prime = 2*np.arccos((radius**2 - cut_radius**2 - radial_positions**2)/(2*radial_positions*radius), where=bool_tpc_radius_intersect)
    area_intersect = theta/2*radius**2 - np.sin(theta)*radius**2/2 + (2*np.pi - theta_prime)/2*cut_radius**2 + np.sin(theta_prime)*cut_radius**2/2
    area_frac = 1-area_intersect/(np.pi*radius**2)
    output[bool_tpc_radius_intersect] = area_frac[bool_tpc_radius_intersect]
    return output

ds_cut = 40
area_norm = get_area_normalization(np.sqrt(np.sum(primary_positions**2, axis=1)), 66.4, ds_cut)
area_cut_bool = ds > ds_cut
SE_weights_with_area_cut = SE_weights/area_norm

SE_hist, SE_errs = plf.histogram_with_weights(np.array(SE_dts)[area_cut_bool], np.array(SE_weights_with_area_cut)[area_cut_bool], histogram_bins)
uncorrelated_SE_rate = np.sum(SE_hist[start_bin:end_bin]*np.diff(histogram_bins)[start_bin:end_bin])
uncorrelated_SE_rate_var = (np.sum((SE_errs[start_bin:end_bin]*np.diff(histogram_bins)[start_bin:end_bin])**2))

import csv
file_path = 'SE_info.csv'
# Write or append data to the CSV file
with open(file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([run, ionization_rate_mean, acceptance, SE_rate, SE_rate_var, uncorrelated_SE_rate, uncorrelated_SE_rate_var])
