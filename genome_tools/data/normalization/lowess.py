from tqdm import tqdm

import pandas as pd
import numpy as np


from scipy.stats import (expon, spearmanr)
from statsmodels.nonparametric import smoothers_lowess

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SEED_NUMBER = 1832245
SEED = np.random.RandomState(SEED_NUMBER)

PEAK_OUTLIER_THRESHOLD = 0.999
DELTA_FRACTION = 0.001
CORRELATION_LIMIT = 0.8
CV_FRACTION = 0.33
SAMPLE_NUMBER = 75000 
BIN_NUMBER = 100


def outlier_limit(x, threshold, k=SAMPLE_NUMBER):
    """
    """

    if len(x) > k:
        subset = x.sample(n=k, replace=False, random_state=SEED)
    else:
        subset = x

    fitted = expon.fit(subset)

    return expon(fitted[0],fitted[1]).ppf(1-(1-threshold)/len(x))
    
def select_uniform(peaks, k=SAMPLE_NUMBER, bins=BIN_NUMBER, top_percentage=PEAK_OUTLIER_THRESHOLD,
                ignore=None, sample_method='raw'):
    """
    Returns row indicies to selected peaks
    """
    
    if ignore is not None:
        p = peaks.loc[peaks.index.difference(ignore)]
    else:
        p = peaks

    if k > len(p):
        k = len(p)

    if sample_method == 'random':
        return p.sample(n=min(len(p), k), replace=False, random_state=SEED).index
        
    if sample_method == 'log':
        vls = np.log(p+1)
    else:
        max_value = outlier_limit(p, top_percentage, k=k)
        keep = (p<max_value)
        vls = p[keep]

    bin_width= (vls.max() - vls.min())/float(bins)
    bin_size = int(np.ceil(k/bins))
    sampled_peaks_indicies = []

    for i in np.arange(vls.min(), vls.max(), bin_width):
        window_min = i
        window_max = i + bin_width

        window_peaks = vls[(vls>=window_min) & (vls<window_max)]

        if len(window_peaks) == 0:
            continue

        sampled_window_peaks = window_peaks.sample(n=min(bin_size, len(window_peaks)), replace=False, random_state=SEED)
        sampled_peaks_indicies = sampled_window_peaks.index.union(sampled_peaks_indicies)

    return sampled_peaks_indicies    

### sets the values out of the range to the edge values.  For 
### logged data, equivalent to linear approximation between 0 and the point
def extrapolate(interpolated, to_extrap, base, predict):
    """
    """

    sample_order = np.argsort(base)
    min_value = np.min(base)
    max_value = np.max(base)
    
    under_min = np.where(to_extrap < min_value)[0]
    over_max = np.where(to_extrap > max_value)[0]
    
    min_predict = predict[sample_order[0]]
    max_predict = predict[sample_order[-1]]
    
    interpolated[under_min] = min_predict
    interpolated[over_max] = max_predict

    return interpolated

def return_interpolated(x, lowess_est, sampled):
    """
    """
    
    # unique, unique_indices = np.unique(peaks[sample],return_index = True)

    interpolated =  np.interp(x, x.loc[sampled], lowess_est.loc[sampled])
    interpolated = extrapolate(interpolated, x.values, x.loc[sampled].values, lowess_est.loc[sampled].values)
    return pd.Series(interpolated, index=x.index)

### choose smoothing parameter for loess by cross-validation
def choose_fraction_cv(x, y, peaks, sampled, start=0.1, end=0.8, step=0.1, delta=None):
    """
    peaks:  peaks in x
    sampled: 
    """
    
    lo = x.loc[sampled].min()
    hi = x.loc[sampled].max()

    within_range = x[(x >= lo) & (x <= hi)].index.intersection(peaks)
    cv_sample = within_range.difference(sampled)
    
    min_err = np.inf
    best_frac = 0

    if delta == None:
        delta = DELTA_FRACTION * np.percentile(x, 99)
    
    for frac in np.arange(start, end+step, step):
        
        smoothed_values = smoothers_lowess.lowess(y.loc[sampled], x.loc[sampled],
                                   return_sorted=False, it=4, frac=frac, delta=delta)
        smoothed_values = pd.Series(smoothed_values, index=sampled)

        interpolated = return_interpolated(x, smoothed_values, sampled)
        
        if pd.isna(interpolated.max()):
            err = np.inf
        else:
            err = ((interpolated.loc[cv_sample]-y.loc[cv_sample])**2).mean()

        if err < min_err:
            min_err = err
            best_frac = frac
    
    return best_frac
        

### normalize all peaks to the geometric mean
### this might need some caution for widely divergent cell types / library sizes
def get_num_samples_per_peak(peaks_mat):
    """
    Returns number of samples for each peak
    """
    return peaks_mat.sum(axis=1)

def scale_matrix(density_mat):
    """
    Returns the the density matrix scaled by the mean
    """
    sample_means = density_mat.mean(axis=0, skipna=True)
    overall_mean = sample_means.mean()
 
    return density_mat.mul(overall_mean/sample_means, axis=1)

def get_pseudocounts(x):
    """
    Compute pseudocounts for each sample in the matrix
    """
    return x.apply(lambda x: x[x>0].min(), axis=0)

def get_geomean(density_mat, psuedocount=True):
    """
    Return geometric mean for each row. We add a small
    pseudocount to each value to avoid buffer underruns.

    """
    if psuedocount:
        pseudocounts = get_pseudocounts(density_mat)
        gm_means = np.log(density_mat.add(pseudocounts, axis=1)).sum(axis=1)
    else:
        gm_means = np.log(density_mat).sum(axis=1)

    gm_means = gm_means / float(density_mat.shape[1])

    return np.exp(gm_means)

def get_geomean_from_peaks(density_mat, peaks_mat):
    """
    """
    counts = peaks_mat.sum(axis=1)

    censored_mat = density_mat.copy()
    censored_mat[~peaks_mat] = np.nan
    censored_mat[censored_mat==0] = np.nan

    gm_means = np.log(censored_mat).sum(axis=1, skipna=True)
    gm_means = np.exp(gm_means * (1/counts))
    
    return gm_means, counts

def get_means(density_mat):
    """
    Returns the mean values across for each row (peaks) across columns (smaples)
    """
    return density_mat.mean(axis=1, skipna=True)

def get_peak_subset(ref_peaks, num_samples_per_peak, density_mat, correlation_limit, min_peak_replication=0.3):
    """
    Select a subset of peaks well correlated to a reference (mean or geometric mean)

    Returns:
    --------
    Indicies for selected subset of peaks
    """
    
    n = num_samples_per_peak.max()

    t = np.floor((np.linspace(.25, 1, 21)[:-1] * density_mat.shape[1]))

    for i in tqdm(t, position=0):
        over = num_samples_per_peak>=i
        avg_cor = np.mean([spearmanr(ref_peaks[over], vals[over])[0]
                    for samp, vals in tqdm(density_mat.iteritems(), total=density_mat.shape[1], position=1, leave=False)])
        print(avg_cor)
        if avg_cor > correlation_limit:
            break

    if i == n:
        print('Caution: individual samples may be poorly captured by mean!') # change to use logger

    return num_samples_per_peak.index[num_samples_per_peak>=i]


def lowess_norm_pair(density1, density2, peaks1, peaks2, pseudocounts1, pseudocounts2, sampled_peaks):
    """
    """
    
    diff = np.log(density2 + pseudocounts2) - np.log(density1 + pseudocounts1)
    
    i = ((density1.loc[sampled_peaks]) > 0 & (density2.loc[sampled_peaks] > 0))
    sampled = i.index[i]

    ref_dens = np.log(density1+pseudocounts1)
    ref_peaks = peaks1[peaks1].index

    cv_fraction = choose_fraction_cv(ref_dens, diff, ref_peaks, sampled_peaks)
    
    smoothed_diff_values = smoothers_lowess.lowess(diff.loc[sampled], ref_dens.loc[sampled],
                                   return_sorted=False, it=4, frac=cv_fraction)
    smoothed_diff_values = pd.Series(smoothed_diff_values, index=sampled)

    interpolated2 = return_interpolated(ref_dens, smoothed_diff_values, sampled)

    return density2  / np.exp(interpolated2)


def lowess_group_norm(density_mat, peaks_mat, sample_number=SAMPLE_NUMBER, 
                        sample_once=False, sample_method='raw', 
                        correlation_limit=CORRELATION_LIMIT):
    """
    Arguments
    ---------
    density_mat: pandas.DataFrame
    peaks_mat: pandas.DataFrame (bool)

    """
    N, S = density_mat.shape

    logger.info('Computing geometric mean for each peak')
    gm_means, num_samples_per_peak = get_geomean_from_peaks(density_mat, peaks_mat)

    pseudocounts = get_pseudocounts(density_mat)

    logger.info('Computing sample distance to \'mean\' dataset')
    dist = np.mean(np.square(density_mat.sub(gm_means, axis=0)), axis=0)
    ref_index = np.argsort(dist)[0]


    if sample_once:
        logger.info('Sampling peaks once')

        decent_peaks = get_peak_subset(gm_means, num_samples_per_peak, density_mat, correlation_limit)
        sampled_peaks = select_uniform(gm_means.loc[decent_peaks], k=sample_number,
                            top_percentage=PEAK_OUTLIER_THRESHOLD,
                            sample_method=sample_method)
        sampled_peaks = decent_peaks.loc[sampled_peaks]


    normed = np.empty(density_mat.shape, dtype=float)
   
    for k in tqdm(range(S), desc='Normalizing samples'):
        if k == ref_index:
            normed[:,k] = density_mat.iloc[:,k]
        else:
            if not sample_once:
                decent_peaks = peaks_mat.iloc[:,k] & peaks_mat.iloc[:, ref_index]
                gm_means = np.exp(np.log(density_mat.iloc[:, k])/2. + np.log(density_mat.iloc[:, ref_index])/2.)
                sampled_peaks = select_uniform(gm_means.loc[decent_peaks], k=sample_number,
                                        top_percentage=PEAK_OUTLIER_THRESHOLD,
                                        sample_method=sample_method)
        
        normed[:,k] = lowess_norm_pair(
                            density_mat.iloc[:, ref_index], 
                            density_mat.iloc[:, k],
                            peaks_mat.iloc[:, ref_index],
                            peaks_mat.iloc[:, k],
                            pseudocounts.iloc[ref_index],
                            pseudocounts.iloc[k],
                            sampled_peaks)                 

    
    return normed


def loess_geomean_norm(density_mat, peaks_mat, sample_number=SAMPLE_NUMBER, 
                       sample_method='random', correlation_limit=CORRELATION_LIMIT,
                       cv_number=5):
    """
    """

    N, S = density_mat.shape

    # numbers = get_peak_numbers(called_vectors)
    num_samples_per_peak = get_num_samples_per_peak(peaks_mat)
    
    peak_means = get_means(density_mat)
    
    #pseudocounts = {d:np.min(size_normed[d][size_normed[d]>0]) for d in size_normed}
    # gm_pseudo = np.mean([pseudocounts[x] for x in pseudocounts])

    pseudocounts = density_mat.apply(lambda x: x[x>0].min(), axis=0)
    gm_pseudo = pseudocounts.mean()
    
    # decent_peaks = get_peak_subset(avg_values,numbers,size_normed,correlation_limit)
    # sampled = select_uniform(avg_values[decent_peaks],number = sample_number,
    #                     sample_method = sample_method)

    decent_peaks = get_peak_subset(peak_means, num_samples_per_peak, density_mat, correlation_limit)

    sampled_peaks = select_uniform(peak_means.loc[decent_peaks], number=sample_number,
                            sample_method=sample_method)


    # sampled = decent_peaks[sampled]
    # xvalues = np.log(avg_values+gm_pseudo)

    log_peak_means = np.log(peak_means + gm_pseudo)
    delta = np.percentile(peak_means, 99) * DELTA_FRACTION

    diffs = np.log(density_mat.add(pseudocounts, axis=1)).sub(log_peak_means, axis=0)
    assert diffs.columns == density_mat.columns

    # delta = np.percentile(avg_values,99)*DELTA_FRACTION
    # difs = {k:np.log(size_normed[k]+pseudocounts[k])-np.log(avg_values+gm_pseudo) for 
    #             k in size_normed}

    ### don't need to cross validate so many times
    

    cv_set = SEED.choice(np.arange(N), size=min(cv_number, N), replace=False)
    cv_fraction = np.mean([choose_fraction_cv(log_peak_means, diffs.iloc[:,i], sampled_peaks, delta=delta) 
                    for i in cv_set])
    
    # cv_set = SEED.choice(size_normed.keys(),size = min(cv_number,len(size_normed.keys())),replace = False)
    # cv_fraction = np.mean([choose_fraction_cv(xvalues,difs[k],sampled,delta = delta)
    #                 for k in cv_set])
    
    normed = np.empty(density_mat.shape, dtype=float)
    for k in tqdm(range(S), desc='Normalizing samples'):
        d = diffs.iloc[:,k]

        smoothed_diff_values =  smoothers_lowess.lowess(d.loc[sampled_peaks], log_peak_means.loc[sampled_peaks],
                                   return_sorted=False, it=4, frac=cv_fraction)

        interpolated2 = return_interpolated(log_peak_means, smoothed_diff_values, sampled_peaks)

        normed[:,k] = density_mat.iloc[:,k] / np.exp(interpolated2)

    # normed_vectors = {}
    # for k in size_normed:
    #     dif = difs[k]
    #     new_values = sm.lowess(dif[sampled],xvalues[sampled],
    #                                    return_sorted=False, it=4, frac = cv_fraction,
    #                                    delta = delta)
    #     interpolated2 = return_interpolated(xvalues, new_values, sampled)
    #     normed_vectors[k] =  size_normed[k]/np.exp(interpolated2)

    return normed
