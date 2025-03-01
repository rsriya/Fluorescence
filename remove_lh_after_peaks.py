import numpy as np
from tqdm import tqdm
import sys
run = sys.argv[1]

import cutax
import strax
st = cutax.xenonnt_online()
st.storage.append(strax.DataDirectory("/scratch/midway3/astroriya/lonehits/", readonly=True))


data_lh = st.get_df(run, ('lone_hits'), keep_columns = ('time', 'channel'))
data_peaks = st.get_df(run, ('peak_basics', 'peak_positions_mlp'))
S2_peaks = data_peaks[data_peaks['type']==2]
import pandas as pd
selected_peaks = pd.read_pickle(f'/scratch/midway3/astroriya/primaries/{run}.pkl')

import power_law_functions as plf
def get_lone_hit_times(selected_peaks, window_lengths, data_lone_hits):
    """
    Extracts the lone hit times, time differences and primary areas for a given set
    of selected peaks and associated window lengths.

    Args:
    - selected_peaks (pandas.DataFrame): a DataFrame with peak information, including
        the 'time' and 'area' columns.
    - window_lengths (list-like): a list or array of window lengths, in seconds,
        associated with each peak.
    - data_lone_hits (pandas.DataFrame): a DataFrame with lone hit information, 
        including the 'time' column.

    Returns:
    - tuple: a tuple of three arrays, containing the lone hit times (in seconds), 
        the time differences with respect to the start of each peak window (in seconds),
        and the primary areas associated with each peak. 

    This function works by iterating over the selected peaks and their associated
    windows, finding all lone hits that fall within each window and extracting their 
    times, time differences and primary areas. The output is returned as a tuple of 
    arrays, with one entry per lone hit found across all windows.
    """
    i = 0
    lone_hit_times = []
    lone_hit_dts = []
    lone_hit_channel = []
    primary_index = []
    N_lh = len(data_lone_hits)
    data_lone_hits_times = data_lone_hits['time'].values
    for row in tqdm(selected_peaks.iterrows(), total=len(selected_peaks)):
        start_time_loop = row[1]['time']
        end_time_loop = row[1]['time'] + window_lengths[i]*1e9
        start_index = 0
        end_index = 0
        start_index, end_index = plf.lone_hit_window_loop_func(data_lone_hits_times, start_time_loop, N_lh, end_time_loop, start_index, end_index)
        this_loop_lonehits = data_lone_hits.iloc[start_index:end_index].query('time < @end_time_loop and time > @start_time_loop')
        lone_hit_times.extend(this_loop_lonehits['time'].values)
        lone_hit_dts.extend(this_loop_lonehits['time'].values - start_time_loop)
        lone_hit_channel.extend(this_loop_lonehits['channel'].values)                
        primary_index.extend([row[0]]*len(this_loop_lonehits))
        i+=1
    return lone_hit_times, lone_hit_dts, lone_hit_channel, primary_index

channel_counts = data_lh['channel'].value_counts()
channel_counts_top = channel_counts[channel_counts.index < 253]
threshold_top = 2*channel_counts_top.median()  # Modify multiplier as needed
channel_counts_bottom = channel_counts[channel_counts.index >= 253]
threshold_bottom = 2*channel_counts_bottom.median()   # Modify multiplier as needed
valid_channels_top = channel_counts_top[channel_counts_top <= threshold_top].index
valid_channels_bottom = channel_counts_bottom[channel_counts_bottom <= threshold_bottom].index
valid_channels = valid_channels_top.union(valid_channels_bottom)
data_lh = data_lh[data_lh['channel'].isin(valid_channels)]

x = 100 * 1e3  # Adjust as needed

# Ensure both DataFrames are sorted by time
data_lh = data_lh.sort_values('time').reset_index(drop=True)
S2_peaks = st.get_df(run, ('peak_basics', 'peak_positions'),keep_columns=('area','time','x', 'y'))
S2_peaks = S2_peaks.sort_values('time').reset_index(drop=True)

# Get time values as numpy arrays
lone_hit_times = data_lh['time'].values
s2_times = S2_peaks['time'].values

# Find the starting and ending indices in data_lone_hits for each S2 time
start_indices = np.searchsorted(lone_hit_times, s2_times)
end_indices = np.searchsorted(lone_hit_times, s2_times + x)

# Create a mask to mark the rows to remove
mask = np.ones(len(data_lh), dtype=bool)

# Set `False` in mask for indices that need to be removed
for start, end in zip(start_indices, end_indices):
    mask[start:end] = False

# Apply mask to filter the DataFrame
data_lh = data_lh[mask]

lone_hit_times, lone_hit_dts, lone_hit_channel, primary_index = get_lone_hit_times(selected_peaks, selected_peaks['window_lengths'].values, data_lh)
del data_lh, S2_peaks
df_sp = pd.DataFrame({
            'time': lone_hit_times,
            'lone_hit_channel': lone_hit_channel,
            'dt_primary':lone_hit_dts,
            'primary_index': primary_index,
        })
df_sp['primary_area']= df_sp['primary_index'].apply(lambda idx: selected_peaks.loc[idx, 'area'])
sorted_windows = np.sort(selected_peaks['window_lengths'].values) #sort window lengths
window_bins_edges = np.concatenate([[0], sorted_windows]) #define windows bins
number_of_overlapping_bins = np.arange(len(window_bins_edges)-1, 0, -1) 
weights = 1/number_of_overlapping_bins #scale bins
histogram_bins = np.logspace(-6, 0.5, 50)    
    
PMT_position = pd.read_csv('/scratch/midway3/astroriya/primaries/pmt_positions_xenonnt.csv')
df_sp['x'] = df_sp['lone_hit_channel'].map(PMT_position.set_index('i')['x'])
df_sp['y'] = df_sp['lone_hit_channel'].map(PMT_position.set_index('i')['y'])
df_sp['array'] = df_sp['lone_hit_channel'].map(PMT_position.set_index('i')['array'])

df_sp_top = df_sp[df_sp['array']=='top']
df_sp_bottom = df_sp[df_sp['array']=='bottom']

start_bin = 28
end_bin = 40
top_lone_hit_weights = weights[np.searchsorted(window_bins_edges, np.array(df_sp_top['dt_primary'])*1e-9, side='right')-1]/np.array(df_sp_top['primary_area'])
weighted_hist, hist_errs = plf.histogram_with_weights(np.array(df_sp_top['dt_primary']), np.array(top_lone_hit_weights), histogram_bins)
top_lh_rate = np.sum(weighted_hist[start_bin:end_bin]*np.diff(histogram_bins)[start_bin:end_bin])
top_lh_rate_var = (np.sum((hist_errs[start_bin:end_bin]*np.diff(histogram_bins)[start_bin:end_bin])**2))

lone_hit_weights = weights[np.searchsorted(window_bins_edges, np.array(df_sp_bottom['dt_primary'])*1e-9, side='right')-1]/np.array(df_sp_bottom['primary_area'])
weighted_hist, hist_errs = plf.histogram_with_weights(np.array(df_sp_bottom['dt_primary']), np.array(lone_hit_weights), histogram_bins)
bottom_lh_rate = np.sum(weighted_hist[start_bin:end_bin]*np.diff(histogram_bins)[start_bin:end_bin])
bottom_lh_rate_var = (np.sum((hist_errs[start_bin:end_bin]*np.diff(histogram_bins)[start_bin:end_bin])**2))

df_sp['primary_x'] = df_sp['primary_index'].apply(lambda idx: selected_peaks.loc[idx, 'x_mlp'])
df_sp['primary_y'] = df_sp['primary_index'].apply(lambda idx: selected_peaks.loc[idx, 'y_mlp'])
df_sp['primary_position'] = np.sqrt(df_sp['primary_x']**2 + df_sp['primary_y']**2)
df_sp['dr'] = np.sqrt((df_sp['x']-df_sp['primary_x'])**2+(df_sp['y']-df_sp['primary_y'])**2)

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
lh_uncorrelated = df_sp[(df_sp['dr']>dr_cut) & (df_sp['array']=='top')]
area_norm = get_area_normalization(lh_uncorrelated['primary_position'].values, 66.4, dr_cut)
lone_hit_weights = weights[np.searchsorted(window_bins_edges, np.array(lh_uncorrelated['dt_primary'])*1e-9, side='right')-1]/np.array(lh_uncorrelated['primary_area'])
lh_weights_with_area_cut = lone_hit_weights/area_norm
weighted_hist, hist_errs = plf.histogram_with_weights(np.array(lh_uncorrelated['dt_primary']), np.array(lh_weights_with_area_cut), histogram_bins)
uncorrelated_lh_rate = np.sum(weighted_hist[start_bin:end_bin]*np.diff(histogram_bins)[start_bin:end_bin])
uncorrelated_lh_rate_var = (np.sum((hist_errs[start_bin:end_bin]*np.diff(histogram_bins)[start_bin:end_bin])**2))

import csv
file_path = 'LH_info_a.csv'
# Write or append data to the CSV file
with open(file_path, mode='a', newline='') as file:
    writer = csv.writer(file)   
    
    writer.writerow([run, threshold_top, threshold_bottom, top_lh_rate, top_lh_rate_var, uncorrelated_lh_rate, uncorrelated_lh_rate_var, bottom_lh_rate, bottom_lh_rate_var])
