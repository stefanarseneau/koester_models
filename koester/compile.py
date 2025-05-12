from scipy.interpolate import interp1d
import numpy as np
import glob
import os

basepath = os.path.dirname(os.path.abspath(__file__))

def read_parameters(file):
    with open(file, 'r') as f:
        data = f.read().splitlines()
    for line in data:
        if line.startswith('TEFF'):
            teff = int(line.split('=')[-1].strip())
        elif line.startswith('LOG_G'):
            log_g = float(line.split('=')[-1].strip())
    return teff, log_g

def read_spectrum(file):
    with open(file, 'r') as f:
        data = f.read().splitlines()
    wavl, flux = [], []
    end_found = False
    for line in data:
        if end_found:
            parts = line.split()
            if len(parts) == 2:  # Ensure it has two values
                wavl.append(float(parts[0]))
                flux.append(float(parts[1]))
        elif line.strip() == 'END':
            end_found = True
    return wavl, flux

def interpolate_onto_best(wavls, fluxes):
    # Find the wavelength grid with the highest resolution
    highest_res_index = np.argmax([len(w) for w in wavls])
    common_wavelengths = wavls[highest_res_index]
    # Interpolate all fluxes onto the common wavelength grid
    interpolated_fluxes = []
    for w, f in zip(wavls, fluxes):
        interp_func = interp1d(w, f, kind='cubic', bounds_error=False)
        interpolated_fluxes.append(interp_func(common_wavelengths))
    return np.array(common_wavelengths), np.array(interpolated_fluxes) * 1e-8

def process_dataset(type = 'DA'):
    assert type in ['DA', 'DB', 'ELM']
    da_files = glob.glob(os.path.join(basepath, type, '*.dk'))
    # read in the parameters from each file
    theta, fluxes, wavls = [], [], []
    for file in da_files:
        theta.append(read_parameters(file))
        wavl, flux = read_spectrum(file)
        fluxes.append(flux)
        wavls.append(wavl)
    theta = np.array(theta)
    # interpolate onto the best grid
    wavl_grid, interp_flux = interpolate_onto_best(wavls, fluxes)
    # save the results
    np.save(os.path.join(basepath, type, 'theta.npy'), theta)
    np.save(os.path.join(basepath, type, 'wavl.npy'), wavl_grid)
    np.save(os.path.join(basepath, type, 'flux.npy'), interp_flux)

def purge_tables(type = 'DA'):
    assert type in ['DA', 'DB', 'ELM']
    for file in glob.glob(os.path.join(basepath, type, '*.npy')):
        os.remove(file)

def check_exists(type):
    if not (os.path.isfile(os.path.join(basepath, type, 'theta.npy')) and os.path.isfile(os.path.join(basepath, type, 'wavl.npy')) 
            and os.path.isfile(os.path.join(basepath, type, 'flux.npy'))):
        print(f"Interpolator files missing for {type}, building now!")
        process_dataset(type = type)

koester_types = ['DA', 'DB', 'ELM']
for type in koester_types:
    check_exists(type)