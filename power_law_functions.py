import numpy as np
from numba import njit, objmode
import pandas as pd
from tqdm import tqdm, trange

@njit
def shadow_area(area, delta_times, overlap_cut_time_s):
    """
    Calculates the shadow area of an object with given area and time delay.

    Parameters:
    area (float): The area of the current peak.
    delta_times (float): The time delay between the current peak and subsequent peaks.
    overlap_cut_time_s (float, optional): The overlap cut time constant in seconds. Default is `overlap_cut_time_s`.

    Returns:
    float: The shadow area of the object.

    """
    return area * np.exp(-delta_times / overlap_cut_time_s)

@njit
def window_loop_func(time_diff_ns, overlap_cut_time_s, overlap_cut_max_times, last_index, current_time, N, primary_times_int):
    """
    This function takes in several parameters including the time difference in nanoseconds, 
    the overlap cut time in seconds, the maximum allowed overlaps, the last index, the current time,
    and the total number of elements. It then iterates through a while loop, incrementing the last index
    and updating the time difference until the time difference exceeds the limit set by the overlap cut time 
    multiplied by the maximum allowed overlaps or the last index becomes greater than or equal to N-2.
    The function returns the updated last index and time difference.
    """
    while time_diff_ns * 1e-9 < overlap_cut_time_s * overlap_cut_max_times and last_index < N-2:
        last_index += 1
        time_diff_ns = primary_times_int[last_index] - current_time
    return last_index, time_diff_ns


def peaks_to_reject(primary_areas, primary_times_int, overlap_cut_time_s, overlap_cut_max_times, overlap_cut_ratio):
    """
    Reject peaks that overlap with other peaks based on their areas.
    
    Args:
        primary_areas (numpy.ndarray): 1D array of peak areas.
        primary_times_int (numpy.ndarray): 1D array of corresponding peak times in nanoseconds.
        overlap_cut_time_s (float): Time constant for exponential overlap cut. Default is `overlap_cut_time_s`.
        overlap_cut_max_times (int): Number of overlap cut time constants to look for. Default is `overlap_cut_max_times`.
        overlap_cut_ratio (float): Ratio of shadow area to primary area that determines if a peak should be rejected. Default is `overlap_cut_ratio`.
    
    Returns:
        numpy.ndarray: 1D array of peak times in nanoseconds that are rejected.
    """
    N = len(primary_areas)
    shadow_areas_arr = np.zeros_like(primary_areas)
    for i in trange(0, N-1):
        current_area = primary_areas[i]
        current_time = primary_times_int[i]
        last_index = i+1
        time_diff_ns = primary_times_int[last_index] - current_time
        
        last_index, time_diff_ns = window_loop_func(time_diff_ns, overlap_cut_time_s, overlap_cut_max_times, last_index, current_time, N, primary_times_int)
        
        shadow_areas_arr[i+1: last_index+1] += shadow_area(current_area, (primary_times_int[i+1: last_index+1] - current_time)*1e-9, overlap_cut_time_s=overlap_cut_time_s)
    rejection_bool = primary_areas*overlap_cut_ratio < shadow_areas_arr
    return np.array(primary_times_int[rejection_bool])

def get_windows(primaries, selected_peaks, overlap_cut_time_s, overlap_cut_max_times, overlap_cut_ratio):
    """
    Computes the window lengths for a set of selected primary peaks.

    Parameters:
    -----------
    primaries: pandas.DataFrame
        A DataFrame containing all primary peaks.
    selected_peaks: pandas.DataFrame
        A DataFrame containing the selected primary peaks.
    overlap_cut_time_s: float, optional (default: overlap_cut_time_s)
        The maximum time overlap between two peaks, in seconds.
    overlap_cut_max_times: int, optional (default: overlap_cut_max_times)
        The maximum number of overlaps allowed between two peaks.

    Returns:
    --------
    window_lengths: list of floats
        A list of window lengths (in seconds) for each peak in selected_peaks.
    """
    window_lengths = []
    N = len(selected_peaks)
    for row in tqdm(selected_peaks.iterrows(), total=N):
        current_time = row[1]['time']
        current_area = row[1]['area']
        primaries_in_shadow = primaries.query('time > @current_time and time < @current_time + @overlap_cut_time_s*@overlap_cut_max_times*1e9')
        delta_times_ns = primaries_in_shadow['time'].values - current_time
        shadow_areas = shadow_area(current_area, delta_times_ns*1e-9, overlap_cut_time_s=overlap_cut_time_s)
        window_end_bool = shadow_areas*overlap_cut_ratio < primaries_in_shadow['area'].values
        peaks_outside_window = delta_times_ns[window_end_bool]
        if len(peaks_outside_window):
            window_lengths.append(np.min(peaks_outside_window)*1e-9)
        else:
            window_lengths.append(0)
    return window_lengths

@njit
def lone_hit_window_loop_func(data_lone_hits_times, start_time_loop, N_lh, end_time_loop, start_index, end_index):
    while data_lone_hits_times[start_index] < start_time_loop and start_index<N_lh-1000:
        start_index += 1000
    start_index -= 1000
    while data_lone_hits_times[end_index] < end_time_loop and start_index<N_lh-1000:
        end_index += 1000
    return start_index, end_index

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
    lone_hit_primary_areas = []
    lone_hit_primary_times = []
    N_lh = len(data_lone_hits)
    data_lone_hits_times = data_lone_hits['time'].values
    for row in tqdm(selected_peaks.iterrows(), total=len(selected_peaks)):
        start_time_loop = row[1]['time']
        end_time_loop = row[1]['time'] + window_lengths[i]*1e9
        start_index = 0
        end_index = 0
        start_index, end_index = lone_hit_window_loop_func(data_lone_hits_times, start_time_loop, N_lh, end_time_loop, start_index, end_index)
        this_loop_lonehits = data_lone_hits.iloc[start_index:end_index].query('time < @end_time_loop and time > @start_time_loop')
        lone_hit_times.extend(this_loop_lonehits['time'].values)
        lone_hit_dts.extend(this_loop_lonehits['time'].values - start_time_loop)
        lone_hit_primary_areas.extend([row[1]['area']]*len(this_loop_lonehits))
        lone_hit_primary_times.extend([start_time_loop]*len(this_loop_lonehits))
        i+=1
    return lone_hit_times, lone_hit_dts, lone_hit_primary_areas, lone_hit_primary_times

@njit
def histogram_with_weights_innerloop(unique_weights, weights_in_bin):
    poisson_numbers = []
    for unique_weight in unique_weights:
        poisson_numbers.append(np.sum(weights_in_bin == unique_weight))
    return poisson_numbers

def histogram_with_weights(items, weights, bins):
    """
    Computes a weighted histogram of a set of items with estimated Gaussian errorbars, using a given set of bin edges.

    Args:
    - items (list-like): a list or array of numerical values to be binned.
    - weights (list-like): a list or array of non-negative weights associated with each item.
    - bins (list-like): a list or array of bin edges (in seconds), defining the boundaries
        of each histogram bin.

    Returns:
    - tuple: a tuple of two arrays, containing the weighted histogram and the corresponding
        errors for each bin.

    This function works by first computing the bin number for each item, based on its value
    and the bin edges provided. It then groups the items and their corresponding weights by 
    bin number, and computes a Poisson error estimate for each unique weight value in each bin,
    using the number of occurrences of that value. Finally, it normalizes the sum of weights 
    within each bin by the width of the bin, and returns the resulting weighted histogram
    and error estimates as separate arrays.

    Note that the input items and the output histogram are assumed to be in units of seconds,
    while the input weights are unitless. The function uses numpy arrays and operations for 
    efficient computation and vectorization.
    """

    N = len(bins) - 1
    bin_widths = np.diff(bins)
    bin_numbers = np.searchsorted(bins, np.array(items)*1e-9, side='right')-1
    weighted_histogram = []
    errors = []
    for i in trange(0, N):
        items_in_bin = items[bin_numbers == i]
        weights_in_bin = weights[bin_numbers == i]
        unique_weights = np.unique(weights_in_bin)
        poisson_numbers = histogram_with_weights_innerloop(unique_weights, weights_in_bin)
        error = np.sqrt(np.sum(poisson_numbers*unique_weights**2))
        summed_weights = np.sum(weights_in_bin)
        weighted_histogram.append(summed_weights)
        errors.append(error)
    return np.array(weighted_histogram)/bin_widths, np.array(errors)/bin_widths

def measure_photoionization_rate_within_drift_time(SE_gain, max_drift_time_s, primaries, S2_peaks):
    """
    This function calculates the photoionization rate within a certain drift time. It takes in four input parameters: 
    the SE gain, the maximum drift time in seconds, a DataFrame of primary scintillation photons, and a DataFrame of
    S2 peaks. The function iterates through each row of the primaries DataFrame, calculating the start and end times for
    the drift time window. It then queries the S2 peaks DataFrame to find all peaks within the window and sums their areas.
    Finally, it divides the summed area by the area of the primary scintillation photon to get the photoionization fraction,
    which is appended to a list. The function returns this list of photoionization fractions.
    """
    N = len(primaries)
    photoionization_frac = []
    for row in tqdm(primaries.iterrows(), total=N):
        start_time = row[1]['time']
        end_time = start_time + max_drift_time_s*1e9
        summed_area = S2_peaks.query('time > @start_time and time < @end_time')['area'].sum()
        photoionization_frac.append(summed_area/row[1]['area'])
    return photoionization_frac

def fit_photoionization_rate(SE_gain, lone_hit_hist, lone_hit_errs, SE_hist, SE_errs, start_bin, hist_bins):
    lone_hit_rate = np.sum(lone_hit_hist[start_bin:]*np.diff(hist_bins)[start_bin:])
    lone_hit_rate_var = (np.sum((lone_hit_errs[start_bin:]*np.diff(hist_bins)[start_bin:])**2))
    SE_rate = np.sum(SE_hist[start_bin:]*np.diff(hist_bins)[start_bin:])
    SE_rate_var = (np.sum((SE_errs[start_bin:]*np.diff(hist_bins)[start_bin:])**2))
    best_fit_ratio = SE_rate/lone_hit_rate*SE_gain
    err = np.sqrt(SE_rate_var/(lone_hit_rate*SE_gain)**2 + lone_hit_rate_var*(SE_rate/(lone_hit_rate**2)*SE_gain)**2)
    return best_fit_ratio, err