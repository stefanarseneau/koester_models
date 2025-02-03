import numpy as np
from scipy.interpolate import RBFInterpolator
import os

basepath = os.path.dirname(os.path.abspath(__file__))

class WDInterpolator:
    def __init__(self, type = 'DA'):
        self.theta = np.load(os.path.join(basepath, type, 'theta.npy'))
        self.fluxes = np.load(os.path.join(basepath, type, 'flux.npy'))
        self.wavl_grid = np.load(os.path.join(basepath, type, 'wavl.npy'))
        self.build_interp_points()

    def build_interp_points(self):
        # normalize the input variables for easier interpolation
        mean_theta, std_theta = np.mean(self.theta, axis=0, keepdims=True), np.std(self.theta, axis=0, keepdims=True)
        theta_norm = (self.theta - mean_theta) / std_theta
        # compute the interpolator function
        def interp(x):
            x_norm = (x - mean_theta) / std_theta # normalize x to pass to the interpolator
            return RBFInterpolator(theta_norm, self.fluxes, kernel='linear')(x_norm)[0] # perform RBF interpolation 
        # return the interpolator function
        self.interpolator = interp