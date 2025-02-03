import numpy as np
from scipy.interpolate import RegularGridInterpolator

import os

basepath = os.path.dirname(os.path.abspath(__file__))

class WDInterpolator:
    def __init__(self, type = 'DA'):
        self.theta = np.load(os.path.join(basepath, type, 'theta.npy'))
        self.fluxes = np.load(os.path.join(basepath, type, 'flux.npy'))
        self.wavl_grid = np.load(os.path.join(basepath, type, 'wavl.npy'))
        self.build_interpolator()

    def build_interpolator(self):
        self.unique_teff = np.array(sorted(list(set(self.theta[:,0]))))
        self.unique_logg = np.array(sorted(list(set(self.theta[:,1]))))
        self.flux_grid = np.zeros((len(self.unique_teff), 
                                len(self.unique_logg), 
                                len(self.wavl_grid)))

        for i in range(len(self.unique_teff)):
            for j in range(len(self.unique_logg)):
                target = [self.unique_teff[i], self.unique_logg[j]]
                try:
                    indx = np.where((self.theta == target).all(axis=1))[0][0]
                    self.flux_grid[i,j] = self.fluxes[indx]
                except IndexError:
                    self.flux_grid[i,j] += -999

        self.model_spec = RegularGridInterpolator((self.unique_teff, self.unique_logg), self.flux_grid) 