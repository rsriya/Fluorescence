import cutax
import strax
st = cutax.xenonnt_online()
st.storage.append(strax.DataDirectory("/project2/lgrandi/xenonnt/processed/", readonly=True))
st.storage.append(strax.DataDirectory("/project/lgrandi/xenonnt/processed/", readonly=True))


from tqdm import trange, tqdm
import numpy as np
import pandas as pd
import sys
import os

run = sys.argv[1]
#run = '053502'
import power_law_functions as plf

max_drift_time_s = 2.3e-3 #s
SE_gain = 31 #https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:abby:segain_evolution

selected_peaks = pd.read_pickle(f'/scratch/midway3/astroriya/primaries/{run}.pkl')

data_peaks = st.get_df(run, ('peak_basics', 'peak_positions_mlp'))
S2_peaks = data_peaks[data_peaks['type']==2]
del data_peaks
S2_peaks = S2_peaks[S2_peaks['tight_coincidence']>=2]
ionization_rates = plf.measure_photoionization_rate_within_drift_time(SE_gain, max_drift_time_s, selected_peaks.query('window_lengths > @max_drift_time_s'), S2_peaks)
ionization_rate_mean = np.mean(ionization_rates)
s2_area_cut = [15,45]
aft_cut = [0.40, 0.99]
width_cut = [90, 650]
SE_data = S2_peaks.query('area < @s2_area_cut[1] and area > @s2_area_cut[0] and area_fraction_top < @aft_cut[1] and area_fraction_top > @aft_cut[0] and range_50p_area < @width_cut[1] and range_50p_area > @width_cut[0]')
del S2_peaks

def get_SE_times(selected_peaks, window_lengths, data_SE):
    i = 0
    SE_times = []
    SE_dts = []
    SE_x = []
    SE_y = []
    SE_cluster = []
    primary_index = []
    N_lh = len(data_SE)
    data_SE_times = data_SE['time'].values
    for row in tqdm(selected_peaks.iterrows(), total=len(selected_peaks)):
        start_time_loop = row[1]['time']
        end_time_loop = row[1]['time'] + window_lengths[i]*1e9
        start_index = 0
        end_index = 0
        start_index, end_index = plf.lone_hit_window_loop_func(data_SE_times, start_time_loop, N_lh, end_time_loop, start_index, end_index)
        this_loop_SE = data_SE.iloc[start_index:end_index].query('time < @end_time_loop and time > @start_time_loop')
        SE_times.extend(this_loop_SE['time'].values)
        SE_dts.extend(this_loop_SE['time'].values - start_time_loop)
        SE_x.extend(this_loop_SE['x_mlp'].values)                
        SE_y.extend(this_loop_SE['y_mlp'].values)                
        SE_cluster.extend(this_loop_SE['cluster'].values)                
        primary_index.extend([row[0]]*len(this_loop_SE))
        i+=1
    return SE_times, SE_dts, SE_x, SE_y, SE_cluster, primary_index
    
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

def generate_signal_events(selected_peak, num_events, alpha, r_range=(0, 66), min_dt=5e6, max_dt=2e8):
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

    phi = np.random.uniform(size=num_events) * 2 * np.pi
    r = r_range[1] * np.sqrt(
        np.random.uniform((r_range[0] / r_range[1]) ** 2, 1, size=num_events)
    )
    signal_events['x_mlp'] = r * np.cos(phi)
    signal_events['y_mlp'] = r * np.sin(phi)
    signal_events['time'] = times
    signal_events['sprinkled'] = True

    return signal_events

num_events = 1#round(np.sum((np.array(SE_dts)>5e6)&(np.array(SE_dts)<4e8))/(2*len(selected_peaks)))

all_signal_events = pd.DataFrame(columns=['x_mlp', 'y_mlp', 'time'])

# Loop through each row in selected_peaks
for _, selected_peak in selected_peaks.iterrows():
    # Generate signal events for the current selected peak
    signal_events = generate_signal_events(selected_peak, num_events, alpha=0.7)
    all_signal_events = pd.concat([all_signal_events, signal_events], ignore_index=True)

# Combine with data_SE
combined_data = pd.concat([SE_data, all_signal_events], ignore_index=True)

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

data_for_clustering = combined_data[['x_mlp', 'y_mlp','time']].values
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_for_clustering)
eps = 0.02
clustering = DBSCAN(eps=eps, min_samples=3, metric='euclidean').fit(data_normalized)
combined_data['cluster'] = clustering.labels_
#combined_data = combined_data[combined_data['cluster'] == -1]
remaining_signal_events = combined_data[(combined_data['cluster'] == -1) & combined_data['sprinkled'].notna()]  # Only the introduced signal events have 'rate' values
acceptance = len(remaining_signal_events)/len(all_signal_events)
SE_data = combined_data[combined_data['sprinkled'].isna()]
#SE_data = SE_data.drop(columns=['cluster'])

SE_times, SE_dts, SE_x, SE_y, SE_cluster, primary_index = get_SE_times(selected_peaks, selected_peaks['window_lengths'].values , SE_data)

df_se = pd.DataFrame({
        'time': SE_times,
        'x': SE_x,
        'y': SE_y,
        'dt_primary': SE_dts,
        'primary_index': primary_index,
        'cluster': SE_cluster
    })
df_se['primary_area']= df_se['primary_index'].apply(lambda idx: selected_peaks.loc[idx, 'area'])
sorted_windows = np.sort(selected_peaks['window_lengths'].values)
window_bins_edges = np.concatenate([[0], sorted_windows])
number_of_overlapping_bins = np.arange(len(window_bins_edges)-1, 0, -1)
weights = 1/number_of_overlapping_bins
histogram_bins = np.logspace(-6, 0.5, 50)

start_bin = 28
end_bin = 40
SE_weights = weights[np.searchsorted(window_bins_edges, np.array(df_se['dt_primary'])*1e-9, side='right')-1]/np.array(df_se['primary_area'])
weighted_hist_SE, hist_errs_SE = plf.histogram_with_weights(np.array(df_se['dt_primary']), np.array(SE_weights), histogram_bins)
SE_rate = np.sum(weighted_hist_SE[start_bin:end_bin]*np.diff(histogram_bins)[start_bin:end_bin])
SE_rate_var = (np.sum((hist_errs_SE[start_bin:end_bin]*np.diff(histogram_bins)[start_bin:end_bin])**2))

df_se['primary_x'] = df_se['primary_index'].apply(lambda idx: selected_peaks.loc[idx, 'x_mlp'])
df_se['primary_y'] = df_se['primary_index'].apply(lambda idx: selected_peaks.loc[idx, 'y_mlp'])
df_se['primary_position'] = np.sqrt(df_se['primary_x']**2 + df_se['primary_y']**2)
df_se['dr'] = np.sqrt((df_se['x']-df_se['primary_x'])**2+(df_se['y']-df_se['primary_y'])**2)

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

dr_cut = 40
se_uncorrelated = df_se[(df_se['dr']>dr_cut)]
area_norm = get_area_normalization(se_uncorrelated['primary_position'].values, 66.4, dr_cut)

se_weights = weights[np.searchsorted(window_bins_edges, np.array(se_uncorrelated['dt_primary'])*1e-9, side='right')-1]/np.array(se_uncorrelated['primary_area'])
se_weights_with_area_cut = se_weights/area_norm
SE_hist, SE_errs = plf.histogram_with_weights(np.array(se_uncorrelated['dt_primary']), np.array(se_weights_with_area_cut), histogram_bins)
uncorrelated_SE_rate = np.sum(SE_hist[start_bin:end_bin]*np.diff(histogram_bins)[start_bin:end_bin])
uncorrelated_SE_rate_var = (np.sum((SE_errs[start_bin:end_bin]*np.diff(histogram_bins)[start_bin:end_bin])**2))

mask = se_uncorrelated['cluster'] == -1
se_uncorrelated_filtered = se_uncorrelated[mask]
area_norm_filtered = area_norm[mask]
se_weights = weights[np.searchsorted(window_bins_edges, np.array(se_uncorrelated_filtered['dt_primary'])*1e-9, side='right')-1]/np.array(se_uncorrelated_filtered['primary_area'])
se_weights_with_area_cut = se_weights/area_norm_filtered
SE_hist, SE_errs = plf.histogram_with_weights(np.array(se_uncorrelated_filtered['dt_primary']), np.array(se_weights_with_area_cut), histogram_bins)
uncorrelated_filtered_SE_rate = np.sum(SE_hist[start_bin:end_bin]*np.diff(histogram_bins)[start_bin:end_bin])
uncorrelated_filtered_SE_rate_var = (np.sum((SE_errs[start_bin:end_bin]*np.diff(histogram_bins)[start_bin:end_bin])**2))

import csv
file_path = 'SE_info.csv'
# Write or append data to the CSV file
with open(file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([run, ionization_rate_mean, acceptance, SE_rate, SE_rate_var, uncorrelated_SE_rate, uncorrelated_SE_rate_var, uncorrelated_filtered_SE_rate, uncorrelated_filtered_SE_rate_var])