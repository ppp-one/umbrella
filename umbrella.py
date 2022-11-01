import pandas as pd
import numpy as np
from scipy.integrate import simps
from scipy.interpolate import griddata
from scipy import ndimage
import matplotlib.pyplot as plt

from IPython.display import clear_output

prePWVGrid = 'prePWVGrid_2400m.pkl'

def interpolate_dfs(index, *data):
    '''
    Interpolates panda dataframes onto an index, of same index type (e.g. wavelength in microns)

    Parameters
    ----------
    index: 1d array which data is to be interpolated onto
    data:       Pandas dataframes 

    Returns
    -------
    df: Interpolated dataframe

    '''
    df = pd.DataFrame({'tmp': index}, index=index)
    for dat in data:
        dat = dat[~dat.index.duplicated(keep='first')]
        df = pd.concat([df, dat], axis=1)
    df = df.interpolate('index').reindex(index)
    df = df.drop('tmp', 1)

    return df

def generateBase(sResponse):
    '''
    Generates the PWV grid base. Takes a few minutes.

    Generates a base grid for:
    airmass: 1 - 3
    pwv: 0.05 - 30 mm
    Teff: 2000 - 36500 K

    See arrays for base resolutions

    Parameters
    ----------
    sResponse:  csv file with two (unlabelled) columns, wavelength (in microns), system spectral response curves of telescope + filter + camera (as fraction).

    Returns
    -------
    coords, data: coordinates and data of base grid generated.
     
    '''
    gridIngredients = pd.read_pickle('./datafiles/' + prePWVGrid)
    rsr = pd.read_csv(sResponse, header=None, index_col=0)

    pwv_values = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0, 30.0])
    airmass_values = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0])
    temperature_values = np.array([2000,  2100,  2250,  2320,  2400,  2440,  2500,  2600,  2650, 2710,  2850,  3000,  3030,  3100,  3200,  3250,  3410,  3500, 3550,  3650,  3700,  3800,  3870,  3940,  4000,  4070,  4190, 4230,  4330,  4410,  4540,  4600,  4700,  4830,  4990,  5040, 5140,  5170,  5240,  5280,  5340,  5490,  5530,  5590,  5660, 5680,  5720,  5770,  5880,  5920,  6000,  6060,  6170,  6240, 6340,  6510,  6640,  6720,  6810,  7030,  7220,  7440,  7500, 7800,  8000,  8080,  8270,  8550,  8840,  9200,  9700, 10400, 10700, 12500, 14000, 14500, 15700, 16700, 17000, 18500, 20600, 24500, 26000, 29000, 31500, 32000, 32500, 33000, 34500, 35000, 36500])

    wavelengths = np.arange(0.5, 2, 0.0001)

    gridSauce = interpolate_dfs(wavelengths, rsr, gridIngredients)
    gridSauce = gridSauce[(gridSauce[1] > 0)]
    atm_grid = []
    for i, pwv in enumerate(pwv_values):
        update_progress(i / (len(pwv_values)-1))
        for airmass in airmass_values:
            for temperature in temperature_values:
                atmosphere_trans = gridSauce[str(pwv) + '_' + str(airmass)] 
                simStar = gridSauce[str(temperature) + 'K']
                response = simps(gridSauce[1]*atmosphere_trans*simStar/max(simStar), gridSauce.index)

                atm_grid.append((pwv, airmass, temperature, response))


    data = np.array([x[3] for x in atm_grid])
    data = data.reshape((len(pwv_values),len(airmass_values),len(temperature_values)))

    coords = np.zeros((len(pwv_values),len(airmass_values),len(temperature_values),3))
    coords[...,0] = pwv_values.reshape((len(pwv_values),1,1))
    coords[...,1] = airmass_values.reshape((1,len(airmass_values),1))
    coords[...,2] = temperature_values.reshape((1,1,len(temperature_values)))

    return coords, data

def interp(coords, data, pwv, airmass, Teff):
    '''
    Interpolates between PWV grid base points (umbrella.generateBase(...)), using cubic method.

    Parameters
    ----------
    coords, data:   coordinates and data of base grid generated.
    pwv:            precipitable water vapour equivalent value at zenith
    airmass:        airmass of star
    Teff:           effective temperature of star

    Returns
    -------
    interp: interpolated value of grid.

    '''
    method = 'cubic' 
    Teffs = coords[..., 2][0,0]
    Teff_lower = np.max(Teffs[Teffs <= Teff])
    Teff_upper = np.min(Teffs[Teffs >= Teff])
    
    if Teff_lower == Teff_upper:
        x = coords[..., 0][coords[..., 2] == Teff] # pwv
        y = coords[..., 1][coords[..., 2] == Teff] # airmass
        z = data[coords[..., 2] == Teff] # effect
        
        interp = griddata((x,y), z, (pwv, airmass), method=method) # interpolated value
    else:
        x_lower = coords[..., 0][coords[..., 2] == Teff_lower] # pwv
        y_lower = coords[..., 1][coords[..., 2] == Teff_lower] # airmass
        z_lower = data[coords[..., 2] == Teff_lower] # effect
        interp_lower = griddata((x_lower,y_lower), z_lower, (pwv, airmass), method=method) # interpolated value lower Teff

        x_upper = coords[..., 0][coords[..., 2] == Teff_upper] # pwv
        y_upper = coords[..., 1][coords[..., 2] == Teff_upper] # airmass
        z_upper = data[coords[..., 2] == Teff_upper] # effect
        interp_upper = griddata((x_upper,y_upper), z_upper, (pwv, airmass), method=method) # interpolated value upper Teff
        
        w_lower = (Teff_upper - Teff) / (Teff_upper - Teff_lower) # lower weight
        w_upper = (Teff - Teff_lower) / (Teff_upper - Teff_lower) # upper weight

        interp = w_lower*interp_lower + w_upper*interp_upper # final interpolated value
    
    return interp

def deltaFluxPlot(coords, data, name=''):
    '''
    Generates a plot of the delta flux with respect to PWV (see Figure 2 in the Pedersen et al. (2022)).

    Parameters
    ----------
    coords, data:   coordinates and data of base grid generated.
    name:           title of plot.

    Returns
    -------
    fig, ax:        figure and axis of plot.

    '''
    cm = plt.cm.get_cmap('RdYlBu')
    airmass = 1
    pwv_values = np.arange(0.05,10.55,0.5)
    temperatures = ticks = [6000,5500,5000,4500,4000,3500,3000,2500,2000]

    eff_df = pd.DataFrame({'pwv': pwv_values})

    fig, ax = plt.subplots()

    for k,temperature in enumerate(temperatures):
        val = interp(coords, data, min(pwv_values), airmass, temperature)
        eff = []
        for pwv in pwv_values:
            val2 = interp(coords, data, pwv, airmass, temperature)
            eff.append(100*(val2/val - 1))

        eff_df[temperature] = eff


    for k,temperature in enumerate(temperatures):
        sc = plt.scatter(eff_df['pwv'],eff_df[temperature], c=np.ones(len(eff_df['pwv']))*temperature, cmap=cm, vmin=min(temperatures), vmax=max(temperatures), marker = 'None', label='_nolegend_')

        ax.plot(eff_df['pwv'], eff_df[temperature], color = sc.to_rgba(temperature))

    cbar = plt.colorbar(sc)

    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels(ticks)

    cbar.set_label(r'T$_{\rm eff}$ [K]', labelpad=20)

    ax.set_xticks(np.arange(0,11,1))
    ax.set_ylim(-10.5, 0.5)
    ax.set_xlabel('PWV [mm]')
    ax.set_ylabel(r'$\Delta$ flux [%]')
    ax.set_title('Name: \'%s\''% (name))
    ax.minorticks_on()
    ax.set_aspect('equal')


    return fig, ax


def getAtmosphere(pwv, airmass, sigma = 0):
    '''
    Returns atmosphere transmission spectrum (from prePWVGrid -- see prePWVGrid generator.ipynb) with a given PWV and airmass.

    Valid PWV and airmass values are:
    pwv_values = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0, 30.0])
    airmass_values = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0])

    Parameters
    ----------
    pwv:            precipitable water vapour equivalent value at zenith
    airmass:        airmass of observation
    sigma:          guassian smoothing of spectrum

    Returns
    -------
    atmosphere_df:     atmosphere spectrum with given PWV and airmass.

    '''

    gridIngredients = pd.read_pickle('./datafiles/' + prePWVGrid)

    if sigma != 0:
        atmosphere = ndimage.gaussian_filter1d(gridIngredients[str(pwv) + '_' + str(airmass)].values, sigma)
        
    atmosphere_df = pd.DataFrame({'transmission' : atmosphere}, index = gridIngredients.index)
        
    return atmosphere_df

def getStar(teff, sigma = 0):
    '''
    Returns stallar spectrum (from prePWVGrid -- see prePWVGrid generator.ipynb) with a given stellar effective temperature.

    Valid PWV and airmass values are:
    temperature_values = np.array([2000,  2100,  2250,  2320,  2400,  2440,  2500,  2600,  2650, 2710,  2850,  3000,  3030,  3100,  3200,  3250,  3410,  3500, 3550,  3650,  3700,  3800,  3870,  3940,  4000,  4070,  4190, 4230,  4330,  4410,  4540,  4600,  4700,  4830,  4990,  5040, 5140,  5170,  5240,  5280,  5340,  5490,  5530,  5590,  5660, 5680,  5720,  5770,  5880,  5920,  6000,  6060,  6170,  6240, 6340,  6510,  6640,  6720,  6810,  7030,  7220,  7440,  7500, 7800,  8000,  8080,  8270,  8550,  8840,  9200,  9700, 10400, 10700, 12500, 14000, 14500, 15700, 16700, 17000, 18500, 20600, 24500, 26000, 29000, 31500, 32000, 32500, 33000, 34500, 35000, 36500])


    Parameters
    ----------
    teff:             effective temperature of star

    Returns
    -------
    star_df:          star spectrum of given stellar effective temperature.

    '''

    gridIngredients = pd.read_pickle('./datafiles/' + prePWVGrid)
    star = gridIngredients[str(teff) + 'K']
    
    if sigma != 0:
        star = ndimage.gaussian_filter1d(gridIngredients[str(teff) + 'K'].values, sigma)
        
    star_df = pd.DataFrame({'spectra': star}, index = gridIngredients.index)
    
    return star_df/star_df.max()

def generateSR(efficiencyFile, filterFile, SRFile):
    '''
    Generates a sensitivity curve for a given telescope and filter.

    Parameters
    ----------
    efficiencyFile:     file containing telescope efficiency curve
    filterFile:         file containing filter transmission curve
    SRFile:             file to save sensitivity curve to

    Returns
    -------
    none - saves sensitivity curve to SRFile

    '''

    wavelengths = np.arange(0.5, 2, 0.0001)

    eff = pd.read_csv(efficiencyFile, header=None)
    filt = pd.read_csv(filterFile, header=None)

    effDF = pd.DataFrame({'eff': eff[1].values}, index=eff[0])
    
    filtDF = pd.DataFrame({'filt': filt[1].values}, index=filt[0])
    
    df = interpolate_dfs(wavelengths, effDF, filtDF)

    dfSR = df['eff']*df['filt']

    dfSR = dfSR[dfSR > 0]

    dfSR.to_csv(SRFile, header=False)
    
    print(SRFile + " has been saved!")

def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)
    
    
def binning(x, y, bins, error=None, std=False, mean_method=np.mean,
            mean_error_method=lambda x: np.sqrt(np.sum(np.power(x, 2))) / len(x)):
    '''
    Binning function, credit to Prose (https://github.com/lgrcia/prose)
    https://github.com/lgrcia/prose/blob/5a5a118e0368d44a5b5b45a46ea710cdc89a30b5/prose/utils.py#L90
    
    '''
    
    bins = np.arange(np.min(x), np.max(x), bins)
    d = np.digitize(x, bins)

    final_bins = []
    binned_flux = []
    binned_error = []
    _std = []

    for i in range(1, np.max(d) + 1):
        s = np.where(d == i)
        if len(s[0]) > 0:
            binned_flux.append(mean_method(y[s[0]]))
            final_bins.append(np.mean(x[s[0]]))
            _std.append(np.std(y[s[0]]) / np.sqrt(len(s[0])))
            if error is not None:
                binned_error.append(mean_error_method(error[s[0]]))

    if std:
        return np.array(final_bins), np.array(binned_flux), np.array(_std)
    elif error is not None and isinstance(error, (np.ndarray, list)):
        return np.array(final_bins), np.array(binned_flux), np.array(binned_error)
    else:
        return np.array(final_bins), np.array(binned_flux)