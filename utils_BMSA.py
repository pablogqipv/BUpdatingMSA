import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, binom, lognorm, ks_2samp
from scipy.optimize import fmin
import random
import pickle
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import csv
import pandas as pd
from os import path
import os
np.random.seed(0)  # to ensure we get the same results

def logL(params, ims, num_col, num_gms):
    """
    Compute the log-likelihood of the data given the parameters.

    Parameters:
    -----------
    params : tuple
        A tuple of two parameters: mean and standard deviation.
    ims : numpy.ndarray
        An array of intensities.
    num_col : numpy.ndarray
        An array of the number of collapses.
    num_gms : numpy.ndarray
        An array of the number of ground motions.

    Returns:
    --------
    float
        The log-likelihood of the data given the parameters.
    """
    prob = norm.cdf(np.log(ims), params[0], params[1])
    prob = np.clip(prob, 1e-10, 1 - 1e-10)
    a = binom.pmf(num_col, num_gms, prob)
    ll = -np.sum(np.log(a))
    return ll

def frag_diff(X, Y1, Y2):
    """
    Compute the difference between fragility curves.

    Parameters:
    -----------
    X : np.ndarray
        X ordinates for both fragility curves.
    Y1 : numpy.ndarray
        Probability values for fragility 1.
    Y2 : numpy.ndarray
        Probability values for fragility 2.

    Returns:
    --------
    np.ndarray
        Values of the difference between functions.
    """
    # Calculate the difference between the two fragility curves
    difference = np.abs(np.array(Y1) - np.array(Y2))
    indices_max_difference = np.argsort(difference)[-3:][::-1]
    intensity_measures_max_difference = [X[i] for i in indices_max_difference]


    # Get the intensity measure at which the fragility curves differ the most
    print(intensity_measures_max_difference)
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlabel('$IM$', fontsize=8)
    ax.set_ylabel("$Difference$", fontsize=8)
    ax.plot(X, difference, 'k-', linewidth=1)
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_mu_prior(mean, dispersion, fig_dir, isave=False, ishow=True):
    """
    Plot the prior probability density function (PDF) of a lognormal distribution.

    Parameters:
    -----------
    mean : float
        The mean (\(\mu\)) parameter of the lognormal distribution.
    dispersion : float
        The dispersion (\(\sigma\)) parameter of the lognormal distribution.
    fig_dir : str
        Directory path to save the figure if `isave` is True.
    isave : bool, optional
        If True, saves the plot as a PNG file in the specified `fig_dir`. Default is False.
    ishow : bool, optional
        If True, displays the plot interactively. Default is True.

    Returns:
    --------
    None
        The function does not return any value. It either shows the plot, saves it, or both.
    """

    # assuming lognormal distribution

    fig, ax = plt.subplots(figsize=(2.7, 2))
    m = mean  # Mean
    s = dispersion  # Standard deviation

    pdf_x = np.linspace(lognorm.ppf(0.001, s, scale=np.exp(np.log(m))),
                        lognorm.ppf(0.999, s, scale=np.exp(np.log(m))), 100)
    pdf_y = lognorm.pdf(pdf_x, s, scale=np.exp(np.log(m)))
    # Plot the PDF
    ax.plot(pdf_x, pdf_y, 'k-', linewidth=1, label="Prior PDF")
    ax.set_ylabel("$Probability$ $Density$", fontsize=8)
    ax.set_xlabel("$\mu$", fontsize=8)

    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlim([0., 4])
    plt.tight_layout()
    if isave:
        plt.savefig(fig_dir + "//" + "mu_prior_Iteration_" + str(1) + ".png", dpi=800)
    if ishow:
        plt.show()
    plt.close()


def plot_sigma_prior(mean, dispersion, fig_dir, isave=False, ishow=True):
    """
    Plot the prior probability density function (PDF) of a lognormal distribution for the dispersion parameter (\(\sigma\)).

    Parameters:
    -----------
    mean : float
        The mean (\(\mu\)) parameter of the lognormal distribution.
    dispersion : float
        The dispersion (\(\sigma\)) parameter of the lognormal distribution.
    fig_dir : str
        Directory path to save the figure if `isave` is True.
    isave : bool, optional
        If True, saves the plot as a PNG file in the specified `fig_dir`. Default is False.
    ishow : bool, optional
        If True, displays the plot interactively. Default is True.

    Returns:
    --------
    None
        The function does not return any value. It either shows the plot, saves it, or both.
    """

    # assuming lognormal distribution

    fig, ax = plt.subplots(figsize=(2.7, 2))
    m = mean  # Mean
    s = dispersion  # Standard deviation

    pdf_x = np.linspace(lognorm.ppf(0.001, s, scale=np.exp(np.log(m))),
                        lognorm.ppf(0.999, s, scale=np.exp(np.log(m))), 100)
    pdf_y = lognorm.pdf(pdf_x, s, scale=np.exp(np.log(m)))
    # Plot the PDF
    ax.plot(pdf_x, pdf_y, 'k-', linewidth=1, label="Prior PDF")
    ax.set_ylabel("$Probability$ $Density$", fontsize=8)
    ax.set_xlabel("$\sigma$", fontsize=8)

    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=7)
    # ax.legend(fontsize=8, frameon=False)
    ax.set_xlim([0., 0.75])
    plt.tight_layout()
    if isave:
        plt.savefig(fig_dir + "//" + "sigma_prior_Iteration_" + str(1) + ".png", dpi=800)
    if ishow:
        plt.show()
    plt.close()


# Log-posterior function
def log_posterior(params, iml, z, num_gms, mu_prior, sigma_prior, mu_prior_std, sigma_prior_std, mean_dist='norm',
                  sigma_dist='norm'):
    """
    Compute the log-posterior probability of the model parameters given the data and priors.

    Parameters:
    -----------
    params : tuple or list
        A pair of model parameters (\(\mu\), \(\sigma\)) for which the posterior is computed.
    iml : numpy.ndarray
        An array of intensity measure levels.
    z : numpy.ndarray
        An array of binary observations (e.g., failure/exceedance).
    num_gms : numpy.ndarray
        An array representing the number of ground motions.
    mu_prior : float
        The mean of the prior distribution for the \(\mu\) parameter.
    sigma_prior : float
        The mean of the prior distribution for the \(\sigma\) parameter.
    mu_prior_std : float
        The standard deviation of the prior distribution for the \(\mu\) parameter.
    sigma_prior_std : float
        The standard deviation of the prior distribution for the \(\sigma\) parameter.
    mean_dist : str, optional
        The type of distribution for the \(\mu\) prior, either 'norm' (normal) or 'log' (lognormal). Default is 'norm'.
    sigma_dist : str, optional
        The type of distribution for the \(\sigma\) prior, either 'norm' (normal) or 'log' (lognormal). Default is 'norm'.

    Returns:
    --------
    float
        The log-posterior probability of the given parameters.
    """

    # Compute prior probability of current or proposed parameters. Joint distribution of two parameters is their product,
    # assuming that they are independent
    if mean_dist == 'log' and sigma_dist == 'log':

        prior = norm.logpdf(params[0], mu_prior, mu_prior_std) + \
                norm.logpdf(params[1], sigma_prior, sigma_prior_std)

    elif mean_dist == 'norm' and sigma_dist == 'norm':
        prior = norm.pdf(params[0], mu_prior, mu_prior_std) + \
                norm.pdf(params[1], sigma_prior, sigma_prior_std)

    # Compute likelihood by multiplying probabilities of each data point
    likelihood = -logL(params, iml, z, num_gms)
    # Compute posterior probability of current or proposed parameters
    posterior = prior + likelihood
    return posterior


# MCMC sampling function
def mcmc(num_samples, init_params, iml, z, num_gms, mu_prior, sigma_prior, mu_prior_std, sigma_prior_std, ppar_mu=0.3, ppar_sigma=0.15):
    """
    Perform Markov Chain Monte Carlo (MCMC) sampling to estimate the posterior distribution of model parameters.

    Parameters:
    -----------
    num_samples : int
        The number of samples to generate in the MCMC chain.
    init_params : list or tuple
        Initial values for the model parameters (\(\mu\), \(\sigma\)).
    iml : numpy.ndarray
        An array of intensity measure levels.
    z : numpy.ndarray
        An array of binary observations (e.g., failure/exceedance).
    num_gms : numpy.ndarray
        An array representing the number of ground motions.
    mu_prior : float
        The mean of the prior distribution for the \(\mu\) parameter.
    sigma_prior : float
        The mean of the prior distribution for the \(\sigma\) parameter.
    mu_prior_std : float
        The standard deviation of the prior distribution for the \(\mu\) parameter.
    sigma_prior_std : float
        The standard deviation of the prior distribution for the \(\sigma\) parameter.
    ppar_mu : float, optional
        The standard deviation of the proposal distribution for \(\mu\) (default is 0.3).
    ppar_sigma : float, optional
        The standard deviation of the proposal distribution for \(\sigma\) (default is 0.15).

    Returns:

    Returns:
    --------
    numpy.ndarray
        An array of sampled parameter values, where each row corresponds to a sample and each column to a parameter.
    """

    samples = []
    current_params = init_params
    # Compute current posterior
    current_posterior = log_posterior(current_params, iml, z, num_gms, mu_prior, sigma_prior, mu_prior_std,
                                      sigma_prior_std)

    for i in range(num_samples):
        random.seed(77)
        # Proposal distribution (Michael Hastings)
        # suggest new parameters, this can be any function
        proposal_params = [np.random.normal(np.exp(current_params[0]), ppar_mu),
                           np.random.normal((current_params[1]), ppar_sigma)]
        lower_bounds = [0.05, 0.05]  # lower bound for the first and second parameter
        upper_bounds = [2, 1]  # upper bound for the first and second parameter
        proposal_params = np.maximum(proposal_params, lower_bounds)
        proposal_params = np.minimum(proposal_params, upper_bounds)
        proposal_params[0] = np.log(proposal_params[0])
        # Compute proposed posterior
        proposal_posterior = log_posterior([(proposal_params[0]), proposal_params[1]], iml, z, num_gms, mu_prior,
                                           sigma_prior, mu_prior_std, sigma_prior_std)
        # Compute acceptance probability as ratio of the posterior densities of the current and proposed parameters
        acceptance_prob = min(1, np.exp(proposal_posterior - current_posterior))

        # Accept or reject proposal
        random.seed(77)
        if np.random.rand() < acceptance_prob:  # update position
            current_params = proposal_params
            current_posterior = proposal_posterior
        samples.append(current_params)

    return np.array(samples)


def autocorr(x, lag):
    """
    Compute the autocorrelation of a sequence at a given lag.

    Parameters:
    -----------
    x : numpy.ndarray
        Input sequence for which the autocorrelation is computed.
    lag : int
        The lag at which to compute the autocorrelation.

    Returns:
    --------
    float
        The autocorrelation coefficient at the specified lag.
    """
    return np.corrcoef(np.array([x[:-lag], x[lag:]]))


def plot_traces(S, fig_dir, iteration=0, ishow=False, isave=True):
    """
    Plot trace plots for the sampled \(\mu\) and \(\sigma\) values from an MCMC simulation.

    Parameters:
    -----------
    S : numpy.ndarray
        Array of sampled parameter values. Each row corresponds to a sample, with the first column representing \(\mu\) (log-transformed)
        and the second column representing \(\sigma\).
    fig_dir : str
        Directory path to save the trace plots.
    iteration : int, optional
        The current iteration index, used in the naming of the saved figures. Default is 0.
    ishow : bool, optional
        If True, displays the plots interactively. Default is False.
    isave : bool, optional
        If True, saves the trace plots as PNG files in the specified `fig_dir`. Default is True.

    Returns:
    --------
    None
        The function does not return any value. It either shows the plots, saves them, or both.
    """

    # Trace of mean
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(np.exp(S[:, 0]), 'k-', linewidth=1)
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlabel('$Samples$', fontsize=8)
    ax.set_ylabel("$\mu$", fontsize=8)
    # ax.legend(fontsize=8, frameon=False)
    ax.set_xlim([0., len(S[:, 0])])
    ax.set_ylim([0.1, 1])
    plt.tight_layout()
    if isave:
        plt.savefig(fig_dir + "//" + "Trace_mean_Iteration_" + str(1 + iteration) + ".png", dpi=800)
    if ishow:
        plt.show()
    plt.close()

    # Trace of sigma
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(S[:, 1], 'k-', linewidth=1)
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlabel('$Samples$', fontsize=8)
    ax.set_ylabel("$\sigma$", fontsize=8)
    # ax.legend(fontsize=8, frameon=False)
    ax.set_xlim([0., len(S[:, 1])])
    ax.set_ylim([0., 1.2])
    plt.tight_layout()
    if isave:
        plt.savefig(fig_dir + "//" + "Trace_sigma_Iteration_" + str(1 + iteration) + ".png", dpi=500)
    if ishow:
        plt.show()
    plt.close()


def plot_ACF(S, fig_dir, iteration=0, ishow=1, isave=1):
    """
    Plot the autocorrelation function (ACF) for the sampled \(\mu\) and \(\sigma\) values from an MCMC simulation.

    Parameters:
    -----------
    S : numpy.ndarray
        Array of sampled parameter values. Each row corresponds to a sample, with the first column representing \(\mu\)
        (log-transformed) and the second column representing \(\sigma\).
    fig_dir : str
        Directory path to save the ACF plot.
    iteration : int, optional
        The current iteration index, used in the naming of the saved figure. Default is 0.
    ishow : bool, optional
        If True, displays the plot interactively. Default is True.
    isave : bool, optional
        If True, saves the ACF plot as a PNG file in the specified `fig_dir`. Default is True.

    Returns:
    --------
    None
        The function does not return any value. It either shows the plot, saves it, or both.
    """
    fig, ax = plt.subplots(figsize=(3, 2))
    lags = range(1, 200)
    autocorrs_m = [autocorr(S[:, 0], lag)[0, 1] for lag in lags]
    ax.plot(lags, autocorrs_m, color='blue', linestyle='solid',
            lw=1, alpha=1, label="$\mu$")
    autocorrs_std = [autocorr(S[:, 1], lag)[0, 1] for lag in lags]
    ax.plot(lags, autocorrs_std, color='red', linestyle='solid',
            lw=1, alpha=1, label="$\sigma$")
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlabel('$Lag$', fontsize=8)
    ax.set_ylabel(r'$Autocorrelation$', fontsize=8)
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim([0., 200])
    ax.set_ylim([-0.025, 1.025])
    plt.tight_layout()
    if isave:
        plt.savefig(fig_dir + "//" + "ACF_Iteration_" + str(1 + iteration) + ".png", dpi=800)
    if ishow:
        plt.show()
    plt.close()


def plot_fragility(sigma_prior, mu_prior, sigma_bench, mu_bench, posterior_std, posterior_mean, nrml_mean,nrml_std, intensity, prob,
                   fig_dir, iml, suffix, iteration=0, ishow=1, isave=1):

    """
    Plot fragility curves and compare prior, posterior, benchmark, and MLE (Maximum Likelihood Estimation) fits against observed data.

    Parameters:
    -----------
    sigma_prior : float
        Standard deviation of the prior distribution for \(\ln(IM)\).
    mu_prior : float
        Mean of the prior distribution for \(\ln(IM)\).
    sigma_bench : float
        Standard deviation of the benchmark distribution.
    mu_bench : float
        Mean of the benchmark distribution.
    posterior_std : float
        Standard deviation of the posterior distribution for \(\ln(IM)\).
    posterior_mean : float
        Mean of the posterior distribution for \(\ln(IM)\).
    nrml_mean : float
        Mean of the normal distribution fit (MLE).
    nrml_std : float
        Standard deviation of the normal distribution fit (MLE).
    intensity : list or numpy.ndarray
        Observed intensity measure (IM) values corresponding to new data points.
    prob : list or numpy.ndarray
        Observed probabilities of exceedance corresponding to the `intensity` values.
    fig_dir : str
        Directory path to save the fragility plot.
    iml : list or numpy.ndarray
        Intensity measure levels used to calculate cumulative probabilities.
    suffix : str
        A suffix to append to the filename of the saved plot.
    iteration : int, optional
        The current iteration index, used in the plot title and filename. Default is 0.
    ishow : bool, optional
        If True, displays the plot interactively. Default is True.
    isave : bool, optional
        If True, saves the fragility plot as a PNG file in the specified `fig_dir`. Default is True.

    Returns:
    --------
    tuple
        - int (list or numpy.ndarray): Intensity measure levels used in calculations.
        - prior_cdf (numpy.ndarray): Cumulative probabilities for the prior distribution at `iml` levels.
        - posterior_cdf (numpy.ndarray): Cumulative probabilities for the posterior distribution at `iml` levels.
    """

    X = np.linspace(0, 5, 500)
    int = iml
    fig, ax = plt.subplots(figsize=(2.2, 2.0))
    # plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
    ax.plot(X, lognorm.cdf(X, sigma_prior, 0, np.exp(mu_prior)), color='mediumturquoise', linestyle='solid', lw=1.2,
            label='Prior', alpha=1)
    ax.plot(X, lognorm.cdf(X, sigma_bench, 0, mu_bench), color='red', linestyle='solid', lw=1.2,
            label='Benchmark', alpha=1)
    ax.plot(X, lognorm.cdf(X, posterior_std, 0, np.exp(posterior_mean)), color='black', linestyle='solid', lw=1.2,
            label='Posterior', alpha=1)
    ax.plot(X, lognorm.cdf(X, nrml_std, 0, np.exp(nrml_mean)), color='green', linestyle='--', lw=1.2,
        label='MLE fit', alpha=1)
    ax.scatter(intensity, prob, edgecolor='black', c='white', marker='o', s=10, alpha=1,
               linewidths=1, label='New data', zorder=3)
    ax.tick_params(axis='y', labelsize=6)
    ax.tick_params(axis='x', labelsize=6)
    ax.set_xlabel('$AvgSA$ $[g]$', fontsize=7)
    ax.set_ylabel(r'$CDF$', fontsize=7)

    ax.set_xlim([0., 1.75])
    ax.set_ylim([-0.025, 1.025])
    # if iteration==4:
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("Iteration:" + str(1 + iteration), fontsize=7)
    plt.tight_layout()
    if isave:
        plt.savefig(fig_dir + "//" + "Fragility_Iteration_" + str(1 + iteration) + suffix +".png", dpi=800)
    if ishow:
        plt.show()
    plt.close()

    return int, lognorm.cdf(int, sigma_prior, 0, np.exp(mu_prior)), lognorm.cdf(int, posterior_std, 0, np.exp(posterior_mean))

def comp_fragility(sigma_prior, mu_prior,  posterior_std, posterior_mean,
                   iml):

    """
    Compute fragility curves for given prior and posterior distributions.

    Parameters:
    -----------
    sigma_prior : float
        Standard deviation of the prior distribution for \(\ln(IM)\).
    mu_prior : float
        Mean of the prior distribution for \(\ln(IM)\).
    posterior_std : float
        Standard deviation of the posterior distribution for \(\ln(IM)\).
    posterior_mean : float
        Mean of the posterior distribution for \(\ln(IM)\).
    iml : list or numpy.ndarray
        Intensity measure levels used to calculate cumulative probabilities.

    Returns:
    --------
    tuple
        - iml (list or numpy.ndarray): Intensity measure levels.
        - prior_cdf (numpy.ndarray): Cumulative probabilities for the prior distribution at `iml` levels.
        - posterior_cdf (numpy.ndarray): Cumulative probabilities for the posterior distribution at `iml` levels.
    """

    X = np.linspace(0, 5, 500)
    int = iml

    return int, lognorm.cdf(int, sigma_prior, 0, np.exp(mu_prior)), lognorm.cdf(int, posterior_std, 0, np.exp(posterior_mean))

def plot_prior(sigma_prior, mu_prior, intensity, prob, fig_dir, ishow=1, isave=1):
    """
    Plot the prior fragility curve and compare it with observed data.

    Parameters:
    -----------
    sigma_prior : float
        Standard deviation of the prior distribution for \(\ln(IM)\).
    mu_prior : float
        Mean of the prior distribution for \(\ln(IM)\).
    intensity : list or numpy.ndarray
        Observed intensity measure (IM) values corresponding to new data points.
    prob : list or numpy.ndarray
        Observed probabilities of exceedance corresponding to the `intensity` values.
    fig_dir : str
        Directory path to save the fragility plot.
    ishow : bool, optional
        If True, displays the plot interactively. Default is True.
    isave : bool, optional
        If True, saves the fragility plot as a PNG file in the specified `fig_dir`. Default is True.

    Returns:
    --------
    None
    """

    X = np.linspace(0, 5, 500)
    fig, ax = plt.subplots(figsize=(2.5, 2.3))
    ax.plot(X, lognorm.cdf(X, sigma_prior, 0, np.exp(mu_prior)), color='mediumturquoise', linestyle='solid', lw=1.2,
            label='Prior', alpha=1)

    ax.scatter(intensity, prob, edgecolor='grey', c='white', marker='o', s=10, alpha=1,
               linewidths=1, label='Data', zorder=3)
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlabel('$AvgSA$ $[g]$', fontsize=8)
    ax.set_ylabel(r'$CDF$', fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.set_xlim([0., 1.75])
    ax.set_ylim([-0.025, 1.025])
    plt.tight_layout()
    if isave:
        plt.savefig(fig_dir + "//" + "Fragility_Iteration_prior.png", dpi=800)
    if ishow:
        plt.show()
    plt.close()


def Gelman(chains):
    """
    Calculate the Gelman-Rubin statistic (R-hat) to assess the convergence of MCMC chains.

    Parameters:
    -----------
    chains : numpy.ndarray
        A 2D array of shape (num_chains, num_samples), where each row represents a separate MCMC chain
        and each column represents a sample from the chain.

    Returns:
    --------
    float
        The potential scale reduction factor (R-hat). Values close to 1 indicate convergence, while
        higher values suggest that further sampling is needed.
    """

    # calculate the within-chain variance
    W = np.mean(np.var(chains, axis=1, ddof=1))

    # calculate the between-chain variance
    B = np.var(np.mean(chains, axis=1), ddof=1)

    # calculate the potential scale reduction factor
    R = np.sqrt((W + B) / W)

    return R

def comp_fragility_reg(duct, Ductility, ims, nIM, nRec):
    """
    Compute the fragility parameters (mean and standard deviation) for a given ductility limit
    and collapse probabilities using regression.

    Parameters:
    -----------
    duct : float
        The ductility threshold for identifying collapse.
    Ductility : numpy.ndarray
        A 2D array where rows represent ground motions, and columns represent intensity measure (IM) levels.
    ims : numpy.ndarray
        Array of intensity measure (IM) levels.
    nIM : int
        Number of intensity measure levels.
    nRec : int
        Number of ground motion records used for each IM level.

    Returns:
    --------
    tuple
        - mean_G1 (float): Logarithmic mean of the IM levels associated with the fragility curve.
        - std_G1 (float): Logarithmic standard deviation of the IM levels associated with the fragility curve.
        - P_collapse (dict): Dictionary containing the collapse probabilities for each IM level.
    """

    im = np.arange(nIM)
    P_collapse = {'G1': []}
    num_col = {'G1': []}

    for i in im:
        temp = Ductility
        mask = temp > duct
        num_col['G1'].append(np.sum(mask[i]) / 1)
        P_collapse['G1'].append(np.sum(mask[i]) / nRec)


    num_gms = nRec * np.ones(nIM)
    params = [np.mean(np.log(ims)), np.std(np.log(ims))]
    FF = fmin(logL, params, args=(ims, num_col['G1'], num_gms))
    mean_G1 = FF[0]
    std_G1 = FF[1]
    print("Mean and std with MLE are:", np.round(np.exp(FF[0]),3), np.round(std_G1,3))

    return mean_G1, std_G1, P_collapse