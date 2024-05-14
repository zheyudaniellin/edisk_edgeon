"""
practice a parabola
y = a (x - b)^2 + c
"""
import pdb
import pickle

# system functions that are always useful to have
import time, sys, os

# basic numeric setup
import numpy as np

# plotting
import matplotlib
from matplotlib import pyplot as plt

# seed the random number generator
rstate= np.random.default_rng(56101)

import dynesty

# truth
a_true = 1.52
b_true = 0.24392
c_true = 1.3043
ndim = 3

# generate mock data
N = 50
x = np.linspace(-5, 5, N)
yerr = 5 + 0.5 * rstate.uniform(size=N)
y_true = a_true * (x - b_true)**2 + c_true
y = y_true + yerr* rstate.normal(size=N)

# plot results
plt.figure(figsize=(10, 5))
plt.errorbar(x, y, yerr=yerr, fmt='ko', ecolor='red')
plt.plot(x, y_true, color='blue', lw=3)
plt.xlabel(r'$X$')
plt.ylabel(r'$Y$')
plt.tight_layout()
plt.show()


def loglike(theta, x, y, yerr):
    a, b, c = theta
    model = a * (x - b)**2 + c
    chisq = np.sum((y - model)**2 / yerr**2)
    return -0.5 * chisq

def prior_transform(utheta):
    ua, ub, uc = utheta
    a = (5 + 2) * ua - 2
    b = (1.5 + 1) * ub - 1
    c = (5 + 5) * uc - 5
    return a, b, c

start_time = time.time()
dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=ndim,
    logl_args=(x, y, yerr), 
    bound='multi', sample='rwalk', rstate=rstate)

dsampler.run_nested(checkpoint_file='testing.pickle')

stop_time = time.time()

dres = dsampler.results
pdb.set_trace()

print('==== Elapsed time = %d seconds ===='%((stop_time-start_time)))
# ==== some quantities ====
from dynesty import utils as dyfunc

# Extract sampling results.
samples = dres.samples  # samples
weights = dres.importance_weights()

# Compute 10%-90% quantiles.
quantiles = [dyfunc.quantile(samps, [0.1, 0.9], weights=weights) for samps in samples.T]

# Compute weighted mean and covariance.
mean, cov = dyfunc.mean_and_cov(samples, weights)

# Resample weighted samples.
# not sure what this means.... 
samples_equal = dres.samples_equal()

# Generate a new set of results with sampling uncertainties.
# not sure what this means....
results_sim = dyfunc.resample_run(dres)

pdb.set_trace()

# ==== plotting ====
from dynesty import plotting as dyplot

# Plot a summary of the run.
rfig, raxes = dyplot.runplot(dres)
plt.show()

# Plot traces and 1-D marginalized posteriors.
# values through each step?
truths = [a_true, b_true, c_true]
labels = [r'$a$', r'$b$', r'$c$']
fig, axes = dyplot.traceplot(dsampler.results, truths=truths, labels=labels,
                             fig=plt.subplots(3, 2, figsize=(16, 12)))
fig.tight_layout()
plt.show()


# corner plot
fig, axes = dyplot.cornerplot(dres, truths=truths, show_titles=True,
                              title_kwargs={'y': 1.04}, labels=labels,
                              fig=plt.subplots(3, 3, figsize=(15, 15)))
plt.show()


# the data points vs fitted profile
x_fit = np.linspace(min(x), max(x), 50)
y_fit = mean[0] * (x_fit - mean[1])**2 + mean[2]
plt.figure(figsize=(10, 5))
plt.errorbar(x, y, yerr=yerr, fmt='ko', ecolor='red', label='data')
plt.plot(x, y_true, color='blue', lw=3, label='truth')
plt.plot(x_fit, y_fit, color='green', lw=3, label='best-fit')
plt.legend()
plt.xlabel(r'$X$')
plt.ylabel(r'$Y$')
plt.tight_layout()
plt.show()

