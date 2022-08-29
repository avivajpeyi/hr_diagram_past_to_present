# -*- coding: utf-8 -*-
# +
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from astroquery.gaia import Gaia
from matplotlib import cm
import numpy as np
import os
from matplotlib import colors
import glob 
 
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

plt.style.use("plot.mplstyle")
pd.set_option('display.max_columns', 500) 



def add_hyg_annotations(ax):
    df = pd.read_csv("data/hyg_hr_theory_data.csv")
    groups = list(set([i.split("_")[0] for i in df.columns.values]))
    for i, g in enumerate(groups[::-1]):
        x, y = df[f"{g}_X"], df[f"{g}_Y"]
        f = interp1d(x, y)
        xnew = np.arange(min(x), max(x), 0.01)
        ynew = gaussian_filter1d(f(xnew), 15, mode="nearest")
        ax.plot(xnew, ynew, alpha=0.7, ls='dashed', c=f"C{i}")
    fs = 13
    ax.annotate("Supergiants", xy=(0.3, -8), xycoords="data", fontsize=fs, )
    ax.annotate("Giants", xy=(0, -1), xycoords="data", fontsize=fs,rotation=-27.5)
    ax.annotate("Subgiants", xy=(1.1,4.5), xycoords="data", fontsize=fs, )
    ax.annotate("Main Sequence", xy=(0, 9.5), xycoords="data", fontsize=fs, rotation=-30)
    ax.annotate("White Dwarfs", xy=(0.3, 15), xycoords="data", fontsize=fs, rotation=-15)

    
def add_1914_annotations(ax):
    x1,y1=[-0.2, 1.5],[2,10]
    m1 = np.diff(y1) / np.diff(x1)
    c1 = y1 - m1 * x1
    x2 = [0, 1.8]
    y2 = m1 * x2 + c1[0] * -0.3
    ax.plot(x1,y1, ls='dashed', color='gray', zorder=-100)
    ax.plot(x2,y2, ls='dashed', color='gray', zorder=-100)



def load_gaia_data():
    i = 0
    step = 50 
    while i < 500:
        file_name = "data/gaia_{}_{}.csv".format(i, i+step)
        if not os.path.isfile(file_name):
            print("Downloading data for: " + file_name)
            job = Gaia.launch_job_async("select top 12000000"
                " bp_rp, phot_g_mean_mag+5*log10(parallax)-10 as mg" 
                " from gaiaedr3.gaia_source"
                " where parallax_over_error > 10"
                " and visibility_periods_used > 8"
                " and phot_g_mean_flux_over_error > 50"
                " and phot_bp_mean_flux_over_error > 20"
                " and phot_rp_mean_flux_over_error > 20"
                " and phot_bp_rp_excess_factor <"
                " 1.3+0.06*power(phot_bp_mean_mag-phot_rp_mean_mag,2)"
                " and phot_bp_rp_excess_factor >"
                " 1.0+0.015*power(phot_bp_mean_mag-phot_rp_mean_mag,2)"
                " and astrometric_chi2_al/(astrometric_n_good_obs_al-5)<"
                "1.44*greatest(1,exp(-0.4*(phot_g_mean_mag-19.5)))"
                + " and 1000/parallax <= {}".format(i + step)
                + " and 1000/parallax > {}".format(i),
                dump_to_file=True,
                verbose=False,
                output_format="csv",
                output_file=file_name)
            r = job.get_results()
        i = i + step
    data = pd.concat(pd.read_csv(f) for f in glob.glob("data/gaia_*.csv"))
    data['temp'] = data.bp_rp.apply(index_to_temp)
    data['ci'] = data['bp_rp']
    return data[['mg','temp','ci']]
    



def load_1914_data():
    # obtained from "A Hertzsprung-Russell diagram for the nineteenth century."
    df = pd.read_csv("data/1914_data.csv")
    df['temp'] = bv_to_temp(df['B - V'])
    df['ci'] = df['B - V']
    return df[['mg','temp', 'ci']]
    

def load_hyg_data():
    data = pd.read_csv("https://raw.githubusercontent.com/astronexus/HYG-Database/master/hygdata_v3.csv")
    data['B-V'] = data['ci']
    data['temp'] = bv_to_temp(data['B-V'])
    data['mg'] = data['absmag']
    data = data[['mg','temp', 'ci']]
    data = data.dropna()
    return data



    
def load_hip_data():
    """
    To download the catalog you must access the VizieR Search Page 
    http://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=I/239/hip_main
    for the catalog I/239 from the Strasbourg astronomical Data Center (CDS).
    Select only the fields listed above: HIP, Vmag, Plx, BV, SpType. 
    In the left pane, under "Preferences" select "unlimited" and "; -Separated Values" 
    Press the "Submit" button and download the file, which I called I_239_selection.tsv 
    and occupies only 3.8 MB. Opening it with a text editor is found that it is a text file 
    with a header of a few lines explanatory of the contents, followed by 118322 entries with 
    the five selected fields separated by the ; character. At the end of the file there is 
    also a blank line we should skip when reading the file.
    
    
    """
    HIP = "data/hip.tsv"
    df = pd.read_table(HIP, skiprows=44, sep='|', header=None, index_col=0,
                   names = ['HIP', 'Vmag', 'Plx', 'B-V', 'SpType'],
                   skipfooter=1, engine='python')
    
    df = df.applymap(lambda x: np.nan if isinstance(x, str)
                           and x.isspace() else x).dropna()

    df['Vmag'] = df['Vmag'].astype(np.float64)
    df['Plx'] = df['Plx'].astype(np.float64)
    df['B-V'] = df['B-V'].astype(np.float64)
    
    # removing nans in parallaxes and B-V
    df = df[ df['Plx'].notna()]
    df = df[df["B-V"].notna()]
    
    #Making sure parallaxes are positive and not too close to zero
    df = df[df.Plx > 0.1] 
    df["d_kpc"] = 1/df["Plx"]
    #     df["d_kpc"].describe()
    

    #creating a copy of the SpTypes without changes to the Spectral Types for later use 
    df['Og_SpType'] = df['SpType']

    #the first 2 identifiers are the most important information in the Spectral types, so we drop all other letters in the column
    df['SpType'] = df['SpType'].str[:2]

    #we also remove spectral types that are < 2 characters, we need a type and a number 
    mask = (df['SpType'].str.len() > 1)
    df = df.loc[mask]

    #we remove spectral types where the second character is not useable, ex: b:, h', mm
    mask = df['SpType'].str[1].str.isdigit()
    df = df.loc[mask]

    #we are only looking at the most common spectral types, O, B A, F, G, K, M, so any other spectral types will be removed 
    df = df[df['SpType'].map(lambda s: s[0] in 'OBAFGKM')]

    #the x axis of the HR-diagram will be the spectral type, so we quantify this by assigning a number to each spectral type
    sequence = {'O':'0', 'B':'1', 'A':'2', 'F':'3', 'G':'4', 'K':'5', 'M':'6'}
    df['SpType'] = df['SpType'].apply(lambda s: sequence[s[0]]+s[1])

    #convert string to float
    df['SpType'].apply(lambda s: float(s))
    
    
    # Add a new column with the absolute magnitude
    df['mg'] = mag_to_absmag(df['Vmag'], df['Plx'])
    df['temp'] = bv_to_temp(df['B-V'])
   
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    df['ci'] = df['B-V']
    return df[['mg','temp', 'ci']]


    
def make_hr_diagram(ax, data, l="", c='k', alpha=0.05, zorder=0, diff_xargs=False, show_colorbar=False, show_ylab=False):
    x,y = data.temp, data.mg

    x = data.ci
    xlab = 'Color Index (G$_{BP}$-G$_{RP}$)'
    xrange = (-1.5, 6.0)

    xscale = 'linear'
    if diff_xargs:
        xlab = "Color Index (B-V)"
        xrange = (-1, 2.5)
    
    
    if len(data) > 1000:
        # only show 2D-histogram for bins with more than 10 stars in them
        norm = colors.PowerNorm(0.5, vmin=0, vmax=20000)
        h = ax.hist2d(x, y, bins=300, cmin=50, norm=norm, zorder=zorder+0.5, cmap='hot', antialiased=True)
        if show_colorbar:
            cb = fig.colorbar(h[3], ax=ax, pad=0.02)
            cb.set_label(r"$\mathrm{Stellar~density}$", labelpad=-25)
            cb.set_ticks([0, 100, 20000])
            cb.set_ticklabels(["1","100", "20K"])
     
    # fill the rest with scatter 
    ax.scatter(x, y, marker='.', alpha=alpha, s=1, color=c, zorder=zorder, rasterized=True)
    
    if l:
        if  not show_colorbar:
            ax.text(0.05, 0.9, l, color='k',ha='left', va='bottom', transform=ax.transAxes)
        else:
            ax.text(0.05, 0.9, l, color='k',ha='left', va='bottom', transform=ax.transAxes)
    
    ax.set_xlabel(xlab)
    if show_ylab:
        ax.set_ylabel('Magnitude (Mv)')   
    ax.set_xscale(xscale)
    ax.set_xlim(*xrange)
    ax.set_ylim(17.5, -15)


def index_to_temp(index):
    if index < -.25:
        index = -.25
    return 5601 / np.power(index + .3, 2.0/3.0)

    
    
def mag_to_absmag(Vmag, Plx):
    return Vmag + 5 * np.log10(Plx/100.)

def bv_to_temp(bv):
    return 4600 * (1/(0.92 * bv + 1.7) + 1/(0.92 * bv + 0.62)) #Ballesteros formula



# -

data_1914 = load_1914_data()
data_gaia = load_gaia_data()
data_hyg = load_hyg_data()

# +
fig, axes = plt.subplots(1,3,  figsize=(12, 5), sharey=True)
make_hr_diagram(axes[0], data_1914,alpha=1,  l="a)",  diff_xargs=True, show_ylab=True)
make_hr_diagram(axes[1], data_hyg, l="b)", diff_xargs=True, alpha=0.1)
make_hr_diagram(axes[2], data_gaia, l="c)", show_colorbar=True)

add_1914_annotations(axes[0])
add_hyg_annotations(axes[1])

plt.tight_layout()
plt.savefig("hr_diagram.png", dpi=300)
