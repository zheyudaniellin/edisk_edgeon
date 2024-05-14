"""
https://github.com/joshspeagle/dynesty/blob/master/demos/Examples%20--%20Linear%20Regression.ipynb


Linear regression is ubiquitous in research. In this example we'll fit a line
    y = m x + b
to data where the error bars have been underestimated and need to be inflated by a factor f. 
"""

# system functions that are always useful to have
import time, sys, os

# basic numeric setup
import numpy as np

# plotting
import matplotlib
from matplotlib import pyplot as plt

# seed the random number generator
rstate= np.random.default_rng(56101)

# re-defining plotting defaults
from matplotlib import rcParams
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'font.size': 30})

import dynesty

# truth
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# generate mock data
N = 50
x = np.sort(10 * rstate.uniform(size=N))
yerr = 0.1 + 0.5 * rstate.uniform(size=N)
y_true = m_true * x + b_true
y = y_true + np.abs(f_true * y_true) * rstate.normal(size=N)
y += yerr * rstate.normal(size=N)

# plot results
plt.figure(figsize=(10, 5))
plt.errorbar(x, y, yerr=yerr, fmt='ko', ecolor='red')
plt.plot(x, y_true, color='blue', lw=3)
plt.xlabel(r'$X$')
plt.ylabel(r'$Y$')
plt.tight_layout()
plt.show()


# log-likelihood
"""
I don't really get what lnf stands for
If I can ignore lnf, then inv_sigma2 is simply 1/noise**2, right? 
The returned argument is then simply chi-squared times -0.5
This is the log-likelihood
P = exp(-0.5 chi^2 ) 
"""
def loglike(theta):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0 / (yerr**2 + model**2 * np.exp(2 * lnf))
    return -0.5 * (np.sum((y-model)**2 * inv_sigma2 - np.log(inv_sigma2)))

# prior transform
"""
utheta is a unit cube. 
each value ranges from 0 to 1
Thus, that determines the range of acceptable values
"""
def prior_transform(utheta):
    um, ub, ulf = utheta
    m = 5.5 * um - 5.
    b = 10. * ub
    lnf = 11. * ulf - 10.
    return m, b, lnf

# Let's sample from this distribution using multiple bounding ellipsoids and random walk
dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim=3,
    bound='multi', sample='rwalk', rstate=rstate)

dsampler.run_nested()
dres = dsampler.results

"""
The values for each parameter is stored in 
> dres.samples
Tt will be an numpy array of nsteps by ndim
Not sure if it's appropriate to call the first dimension "nsteps"

"""
# ==== plotting ====
from dynesty import plotting as dyplot

# values through each step
truths = [m_true, b_true, np.log(f_true)]
labels = [r'$m$', r'$b$', r'$\ln f$']
fig, axes = dyplot.traceplot(dsampler.results, truths=truths, labels=labels,
                             fig=plt.subplots(3, 2, figsize=(16, 12)))
fig.tight_layout()
plt.show()

# corner plot
fig, axes = dyplot.cornerplot(dres, truths=truths, show_titles=True, 
                              title_kwargs={'y': 1.04}, labels=labels,
                              fig=plt.subplots(3, 3, figsize=(15, 15)))
plt.show()

