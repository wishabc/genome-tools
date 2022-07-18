import os

import numpy as np
from numpy import ma
from scipy.stats import expon, spearmanr
from statsmodels.nonparametric import smoothers_lowess
from convert_sparse import convert_to_sparse
import datatable as dt
import multiprocessing as mp
import logging
import sys
import argparse

handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s  %(levelname)s  %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class DataNormalize:
    def __init__(self,
                 peak_outlier_threshold=0.999,
                 delta_fraction=0.001,
                 correlation_limit=0.8,
                 cv_fraction=0.33,
                 seed_number=1832245,
                 sample_number=75000,
                 bin_number=100,
                 jobs=1,
                 ):
        self.seed = np.random.RandomState(seed_number)

        self.sample_number = sample_number
        self.bin_number = bin_number
        self.peak_outlier_threshold = peak_outlier_threshold
        self.delta_fraction = delta_fraction
        self.correlation_limit = correlation_limit
        self.cv_fraction = cv_fraction
        self.scale_factor = None
        self.jobs = mp.cpu_count() if jobs == 0 else jobs

    def outlier_limit(self, x):
        """
        """
        if x.count() > self.sample_number:
            subset = self.sample_masked_array(x, size=self.sample_number)
        else:
            subset = x

        fitted = expon.fit(subset)

        return expon(fitted[0], fitted[1]).ppf(1 - (1 - self.peak_outlier_threshold) / x.count())

    def sample_masked_array(self, arr, size):
        p = ~arr.mask
        return self.seed.choice(np.arange(len(arr)), size=int(size),
                                p=p / p.sum(),
                                replace=False)

    def select_peaks_uniform(self, peaks, decent_indices, ignore=None, sample_method='raw'):
        """
        Returns row indices of selected peaks
        """
        if ignore is not None:
            peaks_mask = decent_indices & ~ignore
        else:
            peaks_mask = decent_indices

        masked_peaks = ma.masked_array(peaks, ~peaks_mask)
        k = min(self.sample_number, masked_peaks.count())

        if sample_method == 'random':
            result = self.sample_masked_array(masked_peaks, k)
        else:
            if sample_method == 'log':
                vls = np.log(masked_peaks + 1)
            elif sample_method == 'raw':
                max_value = self.outlier_limit(masked_peaks)
                new_mask = ~masked_peaks.mask & (masked_peaks < max_value)
                vls = ma.masked_where(~new_mask, masked_peaks)
            else:
                raise ValueError('Method not in (random, log, raw)')
            bin_width = (vls.max() - vls.min()) / self.bin_number
            bin_size = np.ceil(k / self.bin_number)
            sampled_peaks_indicies = []

            for i in np.arange(vls.min(), vls.max(), bin_width):
                window_min = i
                window_max = i + bin_width
                new_mask = ~vls.mask & ((vls >= window_min) & (vls < window_max))
                window_peaks = ma.masked_where(~new_mask, vls)

                if window_peaks.count() == 0:
                    continue
                sampled_window_peaks_indicies = self.sample_masked_array(window_peaks,
                                                                         size=min(bin_size, window_peaks.count()))

                sampled_peaks_indicies.append(sampled_window_peaks_indicies)

            result = np.unique(np.concatenate(sampled_peaks_indicies))
        # Convert to mask
        res = np.zeros(peaks.shape, dtype=bool)
        res[result] = True
        return res

    @staticmethod
    def extrapolate(interpolated, to_extrap, base, predict):
        """
        Sets the values out of the range to the edge values.
        For logged data, equivalent to linear approximation between 0 and the point
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

    def get_extrapolation(self, x, lowess_est, sampled):
        """
        """
        interpolated = np.interp(x, x[sampled], lowess_est)
        extrapolated = self.extrapolate(interpolated, x, x[sampled], lowess_est)
        return extrapolated

    def fit_and_extrapolate(self, y, x, sampled, frac, delta):
        smoothed_values = smoothers_lowess.lowess(y[sampled], x[sampled],
                                                  return_sorted=False, it=4,
                                                  frac=frac, delta=delta)
        return self.get_extrapolation(x, smoothed_values, sampled)

    def choose_fraction_cv(self, y, x, sampled, start=0.1, end=0.8, step=0.1, delta=None):
        """
        choose smoothing parameter for lowess by cross-validation
        sampled:
        """

        lo = x[sampled].min()
        hi = x[sampled].max()

        within_range_indexes = (x >= lo) & (x <= hi)
        cv_sample = within_range_indexes & ~sampled

        min_err = np.inf
        best_frac = 0

        for frac in np.arange(start, end + step, step):
            interpolated = self.fit_and_extrapolate(y, x, sampled, frac, delta)

            if np.isnan(interpolated.max()):
                err = np.inf
            else:
                err = np.power(interpolated[cv_sample] - y[cv_sample], 2).mean()

            if err < min_err:
                min_err = err
                best_frac = frac

        return best_frac

    @staticmethod
    def get_num_samples_per_peak(matrix):
        """
        Returns number of samples for each peak
        """
        return matrix.sum(axis=1)

    @staticmethod
    def get_pseudocounts(matrix):
        """
        Compute pseudocounts for each sample in the matrix
        """
        masked_matrix = ma.masked_equal(matrix, 0.0, copy=False)
        return np.nanmin(masked_matrix, axis=1)

    def get_peak_subset(self, ref_peaks, num_samples_per_peak: np.ndarray, density_mat, correlation_limit,
                        min_peak_replication=0.25) -> np.ndarray:
        """
        Select a subset of peaks well correlated to a reference (mean or geometric mean)

        Returns:
        --------
        Indices for selected subset of peaks
        """

        n = np.max(num_samples_per_peak)

        perc = np.linspace(0, 1, 21)[:-1]
        i = np.where(perc >= min_peak_replication)[0]
        thresholds = np.floor(perc[i] * density_mat.shape[1])

        for i in thresholds:
            over = num_samples_per_peak >= i
            correlations = self.parallel_apply_2D(
                lambda x: spearmanr(ref_peaks[over], x[over]),
                axis=0, arr=density_mat)[0]
            avg_cor = np.mean(correlations)
            if avg_cor > correlation_limit:
                break

        if i == n:
            logger.warning('Caution: individual samples may be poorly captured by mean!')

        return num_samples_per_peak >= i

    def parallel_apply_2D(self, func1d, axis, arr, *args, **kwargs):
        """
        Parallel version of apply_along_axis() for 2D matrices
        """
        other_axis = 1 if axis == 0 else 0
        jobs = min(self.jobs, arr.shape[other_axis])
        if jobs > 1:
            split_arrays = np.array_split(arr, jobs, axis=other_axis)

            ctx = mp.get_context('fork-sever')
            with ctx.Pool(jobs) as p:
                individual_results = p.starmap(np.apply_along_axis(func1d, axis, arr, *args, **kwargs), split_arrays)
            return np.concatenate(individual_results, axis=other_axis)
        else:
            return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

    def lowess_normalize(self, density_mat: np.ndarray, peaks_mat: np.ndarray, cv_numer: int = 5, sample_method='raw'):
        """
        Normalizes to the mean of of the dataset
        Uses only well-correlated peaks to perform normalization
        """
        N, S = density_mat.shape
        logger.info(f'Normalizing matrix with shape: {N:,};{S}')
        num_samples_per_peak = self.get_num_samples_per_peak(peaks_mat)

        logger.info('Computing mean and pseudocounts for each peak')
        pseudocounts = self.get_pseudocounts(density_mat)

        mean_density = density_mat.mean(axis=1)
        mean_pseudocount = pseudocounts.mean()
        xvalues = np.log(mean_density + mean_pseudocount)
        mat_and_pseudo = np.log(density_mat.transpose() + pseudocounts)
        diffs = (mat_and_pseudo - xvalues).transpose()

        logger.info(f'Sampling representative (well-correlated) peaks (r2>{self.correlation_limit}) to mean')
        decent_peaks_mask = self.get_peak_subset(mean_density, num_samples_per_peak, density_mat,
                                                 correlation_limit=self.correlation_limit)
        sampled_peaks_mask = self.select_peaks_uniform(mean_density, decent_peaks_mask,
                                                       sample_method=sample_method)

        logger.info(
            f'Found {decent_peaks_mask.sum():,} well-correlated peaks, using method "{sample_method}"'
            f' and sampled {sampled_peaks_mask.sum():,} peaks')

        logger.info('Computing LOWESS smoothing parameter via cross-validation')
        delta = np.percentile(mean_density, 99) * self.delta_fraction
        cv_set = self.seed.choice(np.arange(S), size=min(cv_numer, S), replace=False)

        cv_fraction = float(
            np.mean(self.parallel_apply_2D(self.choose_fraction_cv, axis=0,
                                           arr=diffs[:cv_set], x=xvalues,
                                           sampled=sampled_peaks_mask, deta=delta)))

        logger.info(f'Computing LOWESS on all the data with params - delta = {delta}, frac = {cv_fraction}')

        norm = self.parallel_apply_2D(self.fit_and_extrapolate, axis=0,
                                      arr=diffs, x=xvalues, sampled=sampled_peaks_mask,
                                      delta=delta, frac=cv_fraction)

        logger.info('Normalizing finished')
        return np.exp(norm)

    @staticmethod
    def get_scale_factor(matrix):
        return 1. / (matrix.sum(axis=0) * 2 / 1.e5)


def check_and_open_matrix_file(path, outpath):
    if not os.path.exists(path):
        raise ValueError(f'{path} do not exist')
    base_path, ext = os.path.splitext(path)
    if ext == '.npy':
        return np.load(path)
    else:
        np_arr = dt.fread(path, header=False).to_numpy()
        np.save(outpath, np_arr)
        return np_arr


def make_out_path(outdir, prefix, matrix_type='signal', mode='sparse'):
    basename = os.path.join(outdir, f'{prefix}.{matrix_type}')
    if mode == 'sparse':
        return basename + '.npz'
    elif mode == 'numpy':
        return basename + '.npy'
    elif mode == 'txt':
        return basename + '.txt'
    else:
        raise ValueError(f'Mode {mode} not in (sparse, numpy, txt)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Matrix normalization using lowess')
    parser.add_argument('peak_matrix', help='Path to binary peaks matrix')
    parser.add_argument('signal_matrix', help='Path to matrix with read counts for each peak in every sample')
    parser.add_argument('output', help='Path to directory to save normalized matrix into.')
    parser.add_argument('--prefix', help='Filenames prefix', default='matrix')
    parser.add_argument('--jobs', type=int,
                        help='Number of jobs to parallelize calculations '
                             '(can\'t be larger than number of samples. If 0 is provided - uses all available cores')
    p_args = parser.parse_args()

    dens_outpath = make_out_path(p_args.output, p_args.prefix, 'signal', 'numpy')
    peaks_outpath = make_out_path(p_args.output, p_args.prefix, 'bin', 'numpy')

    logger.info('Reading matrices')
    counts_matrix = check_and_open_matrix_file(p_args.signal_matrix, dens_outpath)
    peaks_matrix = check_and_open_matrix_file(p_args.peak_matrix, peaks_outpath)

    data_norm = DataNormalize()
    scale_factors = data_norm.get_scale_factor(counts_matrix)
    density_matrix = counts_matrix * scale_factors
    normalizing_matrix = data_norm.lowess_normalize(density_mat=density_matrix, peaks_mat=peaks_matrix)
    r = counts_matrix / normalizing_matrix
    logger.info('Saving results')
    np.savetxt(make_out_path(p_args.output, p_args.prefix, 'normalized', 'txt'), r, delimiter='\t')
    convert_to_sparse(r, make_out_path(p_args.output, p_args.prefix, 'normalized', 'sparse'))