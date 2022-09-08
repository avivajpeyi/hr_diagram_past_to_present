# -*- coding: utf-8 -*-
# %load_ext jupyternotify

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

    
def add_hip_annotations(ax):
    df = pd.read_csv("data/hyg_hr_theory_data.csv")
    groups = list(set([i.split("_")[0] for i in df.columns.values]))
    for i, g in enumerate(groups[::-1]):
        x, y = df[f"{g}_X"], df[f"{g}_Y"]
        if 'dwarfs' in g:
            l = 20
            x, y = x[0:l], y[0:l]
        f = interp1d(x, y)

        xnew = np.arange(min(x), max(x), 0.01)
        ynew = gaussian_filter1d(f(xnew), 15, mode="nearest")
        ax.plot(xnew, ynew, alpha=0.7, ls='dashed', c=f"C{i}")
    fs = 13
    ax.annotate("Giants", xy=(0.1, -1), xycoords="data", fontsize=fs,rotation=-40.)
    ax.annotate("Subgiants", xy=(1.1,3.5), xycoords="data", fontsize=fs, )
    ax.annotate("Main Sequence", xy=(0, 9.), xycoords="data", fontsize=fs, rotation=-40)
    ax.annotate("White Dwarfs", xy=(-0.1, 14.5), xycoords="data", fontsize=fs, rotation=-40)





def load_gaia_data():
    """Inspired by https://github.com/m0zjo-code/Gaia_HR_Plot/blob/main/gaia_HR_plot.py"""
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
    try:
        data = pd.read_csv("data/hyg_data.csv")
    except Exception:
        data = pd.read_csv("https://raw.githubusercontent.com/astronexus/HYG-Database/master/hygdata_v3.csv")
    data['B-V'] = data['ci']
    data['temp'] = bv_to_temp(data['B-V'])
    data['mg'] = data['absmag']
    data = data[['mg','temp', 'ci']]
    data = data.dropna()
    return data
    
def load_hip_data():
    try:
        data = pd.read_csv("data/hyg_data.csv")
    except Exception:
        data = pd.read_csv("https://raw.githubusercontent.com/astronexus/HYG-Database/master/hygdata_v3.csv")
    data['B-V'] = data['ci']
    data['temp'] = bv_to_temp(data['B-V'])
    data['mg'] = data['absmag']
    data = data[['mg','temp', 'ci', 'hip']]
    data = data.dropna()
    return data


    
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



DATA = { "1914":load_1914_data(), 'hyg': load_hyg_data(), 'hip':load_hip_data(), 'gaia':load_gaia_data()}

# -

def plot_hr(datasets=list(DATA.keys()), fname="hr_diagram.png"):
    num = len(datasets)
    data = {k:DATA[k] for k in datasets}
    fig, axes = plt.subplots(1,num,  figsize=(4*num, 5), sharey=True)
    
    for (k, d), ax in zip(data.items(), axes):
        if k == "1914":
            make_hr_diagram(ax, d,alpha=1,  l="a)",  diff_xargs=True, show_ylab=True)
            add_1914_annotations(ax)
        elif k == "hip":
            make_hr_diagram(ax, d, l="b)", diff_xargs=True, alpha=0.1)
            add_hip_annotations(ax)
        elif k == "hyg":
            make_hr_diagram(ax, d, l="bi)", diff_xargs=True, alpha=0.1)
            add_hyg_annotations(ax)
        elif k == "gaia":
            make_hr_diagram(ax, d, l="c)", show_colorbar=True)
    
    
    if 'hyg' not in datasets:  
        for ax in axes:
            ax.set_ylim(17.5, -5)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)

# %%notify
plot_hr(datasets=['1914','hip', 'gaia'], fname="hr_diagram.png")

# %%notify
plot_hr(datasets=['1914','hip', 'hyg', 'gaia'], fname="hr_diagram_all.png")
