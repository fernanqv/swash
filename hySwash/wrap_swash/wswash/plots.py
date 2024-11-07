#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import glob
import shutil

# pip
import numpy as np
import xarray as xr
from scipy import signal as sg
from scipy.signal import hilbert
from scipy.stats import norm, pareto
from scipy import interpolate

# matplotlib import
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib import colors, patches
import matplotlib.animation as animation
import matplotlib.mlab as mlab
#import cmocean

# moviepy
#from moviepy.editor import concatenate_videoclips, ImageClip

import matplotlib as mpl

class SwashPlot(object):
    'SWASH numerical model plotting'

    def __init__(self, swash_proj):
        self.proj = swash_proj

        fig_width_pt = plt.rcParams["figure.figsize"][0] 

        self.plot_params = {'fontsize': 15, 'figure_width': fig_width_pt, 'figure_height': fig_width_pt/3}


    def plot_interpolated(self, ax, xnew, ynew):
        'axes plot interpolated bathymetry'

        # fill axs
        ax.fill_between(
            xnew, - ynew[0],  np.zeros((len(ynew))),
            facecolor = "deepskyblue",
            alpha = 0.5,
            zorder = 1,
        )
        ax.fill_between(
            xnew, np.zeros((len(ynew))) - ynew[0],  -ynew,
            facecolor = "wheat",
            alpha = 1,
            zorder = 2,
        )
        ax.plot(
            xnew,  -ynew,
            color = 'k',
            zorder = 3,
        )

        # customize axes
        ax.set_ylabel('$Depth$ $[m]$', fontweight='bold')
        ax.set_xlabel('$X$ $[m]$', fontweight='bold')
        ax.set_xlim(0, xnew[-1])
        ax.set_ylim(-ynew[0], -min(ynew))

    def plot_computational(self, cc_dx):
        '''
        Plot computational grid and interpolated bathymetry

        cc_dx - computational grid spacing
        '''

        # import depth and spacing values
        depth = np.array(self.proj.depth)
        dx = self.proj.b_grid.dx  # bathymetry spacing
        dxL = self.proj.dxL         # user request nodos/L

        # interpolate bathymetry to computational grid
        x = np.arange(0, len(depth) * dx, dx)
        xnew = np.arange(0, len(depth)*dx, cc_dx)
        xnew = xnew[np.where(xnew <= x[-1])]  # avoid xnew values bellow the interpolation range
        f = interpolate.interp1d(x, depth)
        ynew = f(xnew)
        yp = np.arange(np.min(-depth), np.max(-depth)+cc_dx, cc_dx)

        # plot
        fig, axs = plt.subplots(2, figsize = (12, 8), sharex=True)
        self.plot_interpolated(axs[0], xnew, ynew)
        self.plot_interpolated(axs[1], xnew, ynew)

        # explicit grid
        for h in yp:
            axs[0].plot(
                xnew, np.zeros((len(xnew))) + h,
                color = 'black',
                linestyle = '-',
                linewidth = 0.5,
                zorder = 3,
            )

        for v in xnew:
            axs[0].plot(
                np.zeros((len(yp))) + v, yp,
                color = 'black',
                linestyle = '-',
                linewidth = 0.5,
                zorder = 3,
            )

        axs[0].set_title('$Computational$ $grid$ ${0}$ $nodes/L$'.format(dxL))
        axs[1].set_title('$Interpolated$ $bathymetry$')

    def plot_depthfile(self):
        'Plot bathymetry data including friction or vegetation area in case active commands'

        do_legend = False
        plot_params = self.plot_params
        print(plot_params)

        dx = self.proj.dxinp  # bathymetry spacing
        depth = self.proj.depth
        depth = np.array(depth)

        x = [i * dx for i in range(len(depth))]  

        fig, ax = plt.subplots(1, figsize = (15, 5))

        ax.fill_between(
            x, - depth[0],  np.zeros((len(depth))),
            facecolor = "deepskyblue",
            alpha = 0.5,
            zorder = 1,
        )
        ax.fill_between(
            x, np.zeros((len(depth))) - depth[0],  -depth,
            facecolor = "wheat",
            alpha = 1,
            zorder = 2,
        )
        ax.plot(
            x, -depth,
            color = 'k',
            zorder = 3,
        )

        # friction
        if self.proj.friction_bottom:
            cf_ini = 0
            cf_fin = len(depth) * dx

        elif self.proj.friction_file:
            cf_ini = self.proj.cf_ini * dx
            cf_fin = self.proj.cf_fin * dx

        if self.proj.friction_bottom or self.proj.friction_file:

            pos = [(x >= cf_ini) & (x <= cf_fin)][0]

            # plot
            ax.fill_between(
                x[pos], -depth[pos], -depth[pos] + np.abs(depth[0])/30,
                facecolor = 'crimson',
                alpha = 0.5,
                edgecolor = 'crimson',
                zorder = 3,
                label = 'Friction',
            )
            do_legend = True

        # vegetation
        if self.proj.vegetation != None:

            np_ini = self.proj.np_ini
            np_fin = self.proj.np_fin

            pos = [(x >= np_ini) & (x <= np_fin)][0]

            # plot
            ax.fill_between(
                x[pos], -depth[pos]-np.abs(depth[0]/30), -depth[pos], 
                facecolor = 'green',
                alpha = 0.5,
                edgecolor = 'green',
                zorder = 3,
                label = 'Vegetation',
            )
            do_legend = True

            # TODO: repasar este plot, muy costoso computacionalmente
            #np_ini = self.proj.np_ini
            #np_fin = self.proj.np_fin
            #height = self.proj.height
            #cmap = cmocean.cm.algae_r

            #pos = [(x >= np_ini) & (x <= np_fin)][0]

            #nc = 100

            ## TODO: reduce x ??? no plotea todo
            #x = x[pos]
            #y1 = -depth[pos]
            #y2 = (-depth[pos] + height)

            #z = np.linspace(0, 10, nc)
            #normalize = colors.Normalize(vmin=z.min(), vmax=z.max())

            #for ii in range(len(x)-1):
            #    y = np.linspace(y1[ii], y2[ii], nc)
            #    yn = np.linspace(y1[ii+1], y2[ii+1], nc)
            #    for kk in range(nc - 1):
            #        p = patches.Polygon(
            #            [
            #                [x[ii], y[kk]], [x[ii+1], yn[kk]],
            #                [x[ii+1], yn[kk+1]], [x[ii], y[kk+1]],
            #            ],
            #            color = cmap(normalize(z[kk])),
            #            label = 'Vegetation area',
            #        )
            #        ax.add_patch(p)

            #patch = mpatches.Patch(color='green', alpha=1, label='$Vegetation$', edgecolor=None)

            ## TODO pisa legend de friction
            #ax.legend(handles=[patch], loc='lower right')

        # customize axes
        ax.set_ylabel('$Depth$ $[m]$', fontweight='bold')
        ax.set_xlabel('$X$ $[m]$', fontweight='bold')
        ax.set_xlim(0, x[-1])
        ax.set_ylim(-depth[0], -min(depth)+3)

        if do_legend:
            ax.legend(loc='lower right')

        return fig


    def animate_case_propagation(self, sp, case_id, tini=0, tend=30, tstep=2, figsize=(15,5)):
        
        out = xr.open_dataset(op.join(sp.p_cases, str('{0}'.format(case_id)).zfill(4), 'output.nc'))

        out = out.squeeze()

        plot_params = self.plot_params

        y_depth = - sp.depth 
        x_depth = [i * sp.dxinp for i in range(len(y_depth))]
        y_depth_min = [i * y_depth[0] for i in np.ones((len(x_depth)))]

        fig, ax = plt.subplots(1, figsize=figsize)

        # Init animation
        def init():
            
            return []
        
        # Función de actualización de la animación
        def update(frame):
            ax.clear()

            ax.tick_params(axis='both', which='major', labelsize=plot_params['fontsize'])
            ax.set_xlim(x_depth[0], x_depth[-1])
            ax.set_ylim(y_depth_min[0], y_depth[-1]+3)
            ax.set_xlabel('Cross-shore Distance (m)', fontsize=plot_params['fontsize'])
            ax.set_ylabel('Elevation (m)', fontsize=plot_params['fontsize'])

            # bathymetry
            ax.fill_between(x_depth, y_depth_min, y_depth, fc='wheat', zorder=2)

            # waves
            Y = out.isel(Tsec=frame)['Watlev'].values
            ax.fill_between(range(len(Y)), y_depth_min, Y, fc='deepskyblue', alpha=0.5, zorder=1)

            ax.set_title('Time : {0} s'.format(frame), fontsize=plot_params['fontsize'])
            return []
        
        # Crear animación
        ani = animation.FuncAnimation(fig, update, frames=np.arange(tini, tend, tstep), init_func=init, blit=True)
        plt.close()

        # Mostrar animación
        return ani













    # def plot_waves(self, waves, series):
    #     '''
    #     Plot wave series

    #     waves:   waves parameters
    #     series:  waves series
    #     '''

    #     # waves parameters
    #     WL = waves['WL']
    #     tendc = waves['tendc']
    #     warmup = waves['warmup']
    #     deltat = waves['deltat']

    #     # waves series 
    #     time = series[:,0]
    #     elevation = series[:,1]
    #     envelope = np.abs(hilbert(elevation))  # hilbert transformation

    #     # plot series, group envelope
    #     fig, ax = plt.subplots(1, figsize=(15,3))
    #     ax.plot(time, elevation, c='k')
    #     ax.fill_between(
    #         time, -envelope, envelope,
    #         facecolor = 'dodgerblue',
    #         alpha = 0.3,
    #         label = 'Group envelope',
    #     )
    #     ax.plot(time, envelope, '--', c='k')
    #     ax.plot(time, -envelope, '--', c='k')
    #     ax.set_xlim(0, time[-1])
    #     ax.set_xlabel('$Time$ $[s]$')
    #     ax.set_ylabel('$Elevation$ $[m]$')
    #     ax.legend(loc = 'lower right')

    # def output_gate_spectrum(self, ax, ws, xds_table, gate):
    #     'axes plot Power Spectral Density PSD in (x = gate)'

    #     delttbl = self.proj.delttbl  
    #     warmup = ws['warmup']

    #     wt = xds_table.Tsec.values
    #     wt = xds_table.Tsec.values[np.where(wt >= warmup)[0]]

    #     data = xds_table.isel(Xp = gate)
    #     data = data.sel(Tsec=wt).Watlev.values

    #     # Estimate power spectral density using Welch’s method.        
    #     f, E = sg.welch(data, fs = 1/delttbl , nfft = 512)

    #     # plot
    #     ax.loglog(f, E[0], c='dodgerblue')
    #     ax.grid(which='both', color='lightgray', linestyle='-', linewidth=0.9)
    #     # ax.set_xlim(min(f), max(f))
    #     ax.set_xlim(0,1)
    #     ax.set_ylim(0, 10)
    #     ax.set_xlabel('f [Hz]', fontweight='bold')
    #     # ax.yaxis.set_label_position("right")
    #     ax.set_ylabel(r'PDS $[m^{2} s]$', fontweight='bold')

    # def output_gate_series(self, ax, ws, xds_table, gate):
    #     'axes plot watlev for (x = gate)'

    #     warmup = ws['warmup']
    #     H = ws['H']

    #     wt = xds_table.Tsec.values
    #     wt = xds_table.Tsec.values[np.where(wt >= warmup)[0]]

    #     data = xds_table.isel(Xp = gate)
    #     data = data.sel(Tsec=wt).Watlev.values

    #     # Water level series
    #     ax.plot(xds_table.Tsec.values, xds_table.isel(Xp = gate).Watlev.squeeze().values, c='k')
    #     ax.set_xlim(xds_table.Tsec.values[0], xds_table.Tsec.values[-1])
    #     ax.set_ylabel('\u03B7 $[m] (WG = {0})$'.format(gate), fontweight='bold')
    #     ax.set_xlabel('Time [m]', fontweight='bold')
    #     ax.set_ylim(-1.5*np.float(H), 1.5*np.float(H))

    # def output_forcing(self, ax, x, xds_table, p, t):
    #     'axes plot wave forcing in the west boundary inluding spinup time'

    #     # read vars
    #     wt = xds_table.Tsec.values
    #     waves = xds_table.sel(Xp=x, Tsec=wt).Watlev.values.squeeze()
    #     pw = np.where(xds_table.isel(Tsec=p).Watlev.values!=-99)

    #     # set axis to fill between
    #     y1 = -xds_table.sel(Tsec=t).Botlev.values[pw[0]]
    #     y2 = xds_table.sel(Tsec=t).Watlev.values[pw[0]]

    #     y1 = np.reshape(y1,(len(y1,)))
    #     y2 = np.reshape(y2,(len(y2,)))

    #     # customize axes          
    #     ax.plot(wt, waves, 'k', linewidth=1, zorder=1)
    #     ax.scatter(wt[p], waves[p], c='red', s=20, marker='o', zorder=2)
    #     ax.set_ylabel('\u03B7 $[m] (x = {0})$'.format(x), fontweight='bold', horizontalalignment='left')
    #     ax.set_xlim(wt[0], wt[-1])

    # def output_profile(self, ax, post, xds_table, t, depth):
    #     'axes plot watlev from spaced stations'

    #     h0 = post['h0']

    #     pw = np.where(xds_table.sel(Tsec=t).Watlev.values != -99.0)  
    #     x = xds_table.Xp.values[pw[0]]

    #     # Set axis to fill between
    #     y1 = -xds_table.sel(Tsec=t).Botlev.values[pw[0]]
    #     y2 = xds_table.sel(Tsec=t).Watlev.values[pw[0]]

    #     y1 = np.reshape(y1,(len(y1,)))
    #     y2 = np.reshape(y2,(len(y2,)))

    #     # water level
    #     ax.fill_between(
    #         x, y1, y2,
    #         facecolor = "deepskyblue",
    #         alpha = 0.5,
    #     )
    #     ax.fill_between(
    #         xds_table.Xp.values, np.ones((len(xds_table.Xp.values)))*-h0,
    #         -xds_table.sel(Tsec=t).Botlev.squeeze().values,
    #         facecolor = "wheat",
    #         alpha = 1,
    #     )
    #     ax.plot(xds_table.Xp.values, -xds_table.sel(Tsec=t).Botlev.values, color = 'k')

    #     # customize plot
    #     ax.set_ylabel('y [m]', fontweight='bold')
    #     ax.set_xlim(xds_table.Xp.values[0], xds_table.Xp.values[-1])
    #     ax.set_ylim(-h0, np.nanmax(-xds_table.sel(Tsec=0).Botlev.values)+2)

    # def output_gate_runup(self, ax, post, xds_table, p, t):
    #     ' axes plot output runup from intersection free surface-profile'

    #     ru2 = post['ru2']

    #     # read vars
    #     wt = xds_table.Tsec.values
    #     runup = xds_table.sel(Tsec=wt).Runlev.values*100

    #     # runrup
    #     ax.plot(wt, runup, 'b', linewidth=1, zorder=1)
    #     # ax.scatter(wt[p], runup[p], c='r', s=25, marker='o', zorder=2)
    #     ax.set_xlim(wt[0], wt[-1])
    #     ax.set_ylabel('Ru $[cm]$', fontweight='bold', horizontalalignment='left')

    #     # Validation box
    #     props = dict(
    #         boxstyle = 'round',
    #         facecolor = 'azure',
    #         alpha = 1,
    #         edgecolor = 'lightcyan',
    #     )
    #     if np.isnan(runup).all() == True:
    #         ax.text(
    #             0.03, 0.8, '$No$ $runup$ $computed$',
    #             transform = ax.transAxes,
    #             fontsize = 10,
    #             horizontalalignment = 'left',
    #             bbox = props,
    #         )
    #     else:
    #         ax.text(
    #             0.03, 0.8, 'Ru2% = {0:.2f} (m)'.format(np.float(ru2)),
    #             transform = ax.transAxes,
    #             fontsize = 10,
    #             horizontalalignment = 'left',
    #             bbox = props,
    #         )

    # def output_gate_disch(self, ax, post, xds_table, p, t):
    #     'axes plot output instantaneous discharge magnitude'

    #     dx = self.proj.b_grid.dx
    #     gate = int(post['Gate_Q'])
    #     q = post['q']

    #     # read vars
    #     wt = xds_table.Tsec.values
    #     disch = xds_table.sel(Tsec=wt, Xp=gate).Qmag.squeeze().values

    #     # Discharge
    #     ax.plot(wt, disch, 'crimson', linewidth=1, zorder=1)
    #     ax.scatter(wt[p], disch[p], marker='o', c='b', s=25, zorder=2)
    #     ax.set_xlim(wt[0], wt[-1])
    #     ax.set_ylim(0, max(disch)+0.1)
    #     ax.set_ylabel('Q $[m^{2}/s]$', fontweight='bold')

    #     # Validation box
    #     props = dict(
    #         boxstyle='round',
    #         facecolor='azure',
    #         alpha=1,
    #         edgecolor='lightcyan',
    #     )

    #     if np.isnan(q).all() == True:
    #         ax.text(
    #             0.03, 0.8, '$No$ $overtopping$ $computed$'.format(np.float(q)),
    #             transform = ax.transAxes,
    #             fontsize = 10,
    #             horizontalalignment = 'left',
    #             bbox = props,
    #         )
    #     else:
    #         ax.text(
    #             0.03, 0.8, '$q$ $Swash$ $=$ ${0:.2f}$ $[m/l/s]$'.format(np.float(q)),
    #             transform = ax.transAxes,
    #             fontsize = 10,
    #             horizontalalignment = 'left',
    #             bbox = props,
    #         )

    # def output_contribution(self, ax, df_Hi):
    #     'axes plot energy contribution of different frequency waves (IC, IG, VLF) measured by Hs2'

    #     ic = df_Hi.HIC.values
    #     ig = df_Hi.HIG.values
    #     vlf = df_Hi.HVLF.values

    #     ic[np.where(np.isnan(ic)==True)] = 0
    #     ig[np.where(np.isnan(ig)==True)] = 0
    #     vlf[np.where(np.isnan(vlf)==True)] = 0

    #     ax.fill_between(
    #         df_Hi.Xi.values, ic**2,
    #         color = 'yellowgreen',
    #         edgecolor = None,
    #         alpha = 0.6,
    #     )
    #     ax.fill_between(
    #         df_Hi.Xi.values, ic**2, ic**2+ig**2,
    #         color = 'navy',
    #         edgecolor = None,
    #         alpha = 0.5,
    #     )
    #     ax.fill_between(
    #         df_Hi.Xi.values, ic**2+ig**2, ig**2+ic**2+vlf**2,
    #         edgecolor = None,
    #         color = 'navy',
    #         alpha = 1,
    #     )

    #     # customize axes
    #     patchIC = mpatches.Patch(color='yellowgreen', alpha=0.6, label='$H_{IC}^{2}$ $[{m}^{2}]$', edgecolor=None)
    #     patchIG = mpatches.Patch(color='navy', alpha=0.5, label='$H_{IG}^{2}$ $[{m}^{2}]$', edgecolor=None)
    #     patchVLF = mpatches.Patch(color='navy', alpha=1, label='$H_{VLF}^{2}$ $[{m}^{2}]$', edgecolor=None)
    #     ax.legend(handles=[patchIC, patchIG, patchVLF], loc='lower left')
    #     ax.set_xlim(df_Hi.Xi.values[0], df_Hi.Xi.values[-1])
    #     ax.set_ylim(0, max(ig**2+ic**2+vlf**2)+0.5)

    # def output_fft(self, ax, df_Hi):
    #     'axes plot output fourier transformation Hi, Hs, Hmax'

    #     ax.plot(df_Hi.Xi.values, df_Hi.HIC.values, c='r', label='$H_{IC}$')
    #     ax.plot(df_Hi.Xi.values, df_Hi.HIG.values, c='k', label='$H_{IG}$')
    #     ax.plot(df_Hi.Xi.values, df_Hi.HVLF.values, '--', c='k', label='$H_{VLF}$')

    #     # customize axes
    #     ax.set_xlim(df_Hi.Xi.values[0], df_Hi.Xi.values[-1])
    #     ax.legend(loc ='lower left')

    # def output_env(self, ax, ws, post, xds_table, df, df_Hi, depth):
    #     'axes plot maximun and minimun free surface'

    #     ru2 = post['ru2']
    #     h0 = post['h0']
    #     gate = post['Gate_Q']
    #     kr = post['kr']

    #     WL = ws['WL']

    #     hini = -xds_table.isel(Tsec=0).Botlev.values[0]
    #     dx = self.proj.b_grid.dx

    #     # Set axis to fill between
    #     x = xds_table.Xp.values[np.where(xds_table.Xp.values <= df_Hi.Xi.values[-1])]
    #     y = -xds_table.Botlev.isel(Tsec=0).values[np.where(xds_table.Xp.values <= df_Hi.Xi.values[-1])[0]]
    #     y = np.reshape(y,(len(y),))

    #     # Interpolate between bathymetry values
    #     f = interpolate.interp1d(x, y, kind='linear')
    #     ynew = f(df_Hi.Xi)

    #     # water level
    #     ax.fill_between(
    #         df_Hi.Xi, df_Hi.E_min, df_Hi.E_max,
    #         facecolor = "deepskyblue",
    #         alpha = 1,
    #         label = '\u03B7 [m]',
    #     )
    #     ax.fill_between(
    #         df_Hi.Xi, ynew, df_Hi.E_min,
    #         facecolor = 'deepskyblue',
    #         alpha = .5,
    #     )
    #     ax.fill_between(
    #         xds_table.Xp.values, np.zeros((len(xds_table.Xp.values)))+(hini),
    #         -xds_table.isel(Tsec=0).Botlev.squeeze().values,
    #         facecolor = "wheat",
    #         alpha=1,
    #     )
    #     ax.plot(xds_table.Xp.values, -xds_table.isel(Tsec=0).Botlev.values, color = 'k')

    #     # mean setup
    #     ax.plot(df.Xp.values, df.Setup.values, c='blue', zorder=1, label='Setup')

    #     # runup elevation point
    #     # if np.isnan(ru2) == False:
    #     # idx1 = np.argwhere(np.diff(np.sign(depth - np.zeros(len(depth))+ru2-WL)) != 0).reshape(-1) + 0
    #     # ax.scatter(range(len(depth))[idx1[0]], ru2-WL, marker='o', s= 20, c='coral', alpha=1, zorder=3, label='$Ru_{2}$')

    #     # Gate to overtopping
    #     ax.scatter(
    #         gate, -depth[int(gate/dx)],
    #         marker = 'o',
    #         s = 20,
    #         c = 'coral',
    #         alpha = 1,
    #         zorder = 3,
    #         label='Overtopping Gate',
    #     )

    #     # customize axes
    #     ax.set_ylabel('y [m]', fontweight='bold')
    #     #ax.set_xlim(xds_table.Xp.values[0], xds_table.Xp.values[-1])
    #     ax.set_xlim(xds_table.Xp.values[0], xds_table.Xp.values[-1])
    #     ax.set_ylim(-h0, np.nanmax(-xds_table.sel(Tsec=0).Botlev.values)+2)
    #     ax.legend(loc='lower right')
    #     props = dict(
    #         boxstyle = 'round',
    #         facecolor = 'snow',
    #         alpha = 0.6,
    #         edgecolor = 'w',
    #     )
    #     ax.text(
    #         0.02, 0.05, '  $Kr = {0:.2f}$ \n $h0 = {1:.2f} m$ \n  $dx = {2:.2f}$'.format(
    #             np.float(kr), np.float(h0), np.float(dx)
    #         ),
    #         transform = ax.transAxes,
    #         fontsize=10,
    #         bbox=props,
    #     )

    # def hist_Hi(self, ax, df_Hi, ds_fft_hi, wg, x, num_bins):
    #     'Mendez et al. 2014 Transformation model of wave heigh distribution'

    #     Hi = ds_fft_hi.sel(Xp=wg).Hi.values
    #     H = df_Hi.iloc[x].H
    #     Hmax = np.nanmax(Hi)
    #     Hrms = np.sqrt((1/len(Hi))*np.sum(Hi**2))
    #     H13 = np.percentile(Hi, 67)

    #     # shape parameter kappa
    #     kappa = Hrms / Hmax
    #     fi = (1-kappa**0.944)**1.187

    #     # the histogram of the data
    #     n, bins, patches = ax.hist(Hi, num_bins, color='yellowgreen', alpha=0.9, density=1, rwidth=0.95, zorder=2)
    #     bins = np.linspace(0, bins[-1]+0.5, 100)

    #     # Transformation of wave heigh distribution (Rayleigh kappa = 0)
    #     #y = 4.01*(bins/Hs**2)*np.exp(-2.005*((bins**2)/(Hs**2)))
    #     y = ((2*(fi**2)*Hrms*bins)/(Hrms-kappa*bins)**3)*np.exp(-(fi**2)*(bins/(Hrms-kappa*bins))**2)

    #     ax.plot(bins, y, '-', c='k', label='Kappa = {0}'.format(kappa))
    #     ax.set_xlabel('$Normalized$ $wave$ $heigh$ $(m)$')
    #     ax.set_ylabel('$Probability$ $density$')
    #     ax.set_title('Wave heigh distribution X = WG{0}'.format(x), fontweight='bold')

    #     ax.legend(loc='upper right')

    #     props = dict(
    #         boxstyle = 'round',
    #         facecolor = 'cyan',
    #         alpha = 0.1,
    #         edgecolor = 'w',
    #     )
    #     ax.text(
    #         0.75, 0.75, '$H%$ $=$ ${0:.2f}$ $m$\n$H_13$ $=$ ${1:.2f}$ $m$'.format(np.float(H), H13),
    #         transform = ax.transAxes,
    #         fontsize = 10,
    #         bbox = props,
    #     )

    # def hist_Ru(self, ax, ax1, ax2, ru2, g, num_bins):
    #     ax1.axis('off')
    #     ax2.axis('off')

    #     # the histogram of the data
    #     n, bins, patches = ax.hist(
    #         g, num_bins,
    #         color = 'orangered',
    #         alpha = 0.9,
    #         rwidth = 0.92,
    #         density = 1,
    #         zorder = 2,
    #     )

    #     ax.set_xlabel('$Normalized$ $runup$')
    #     ax.set_ylabel('$Probability$ $density$')
    #     ax.set_title('Runup distribution', fontweight='bold')

    #     props = dict(
    #         boxstyle = 'round',
    #         facecolor = 'orangered',
    #         alpha = 0.10,
    #         edgecolor = 'w'
    #     )
    #     ax.text(
    #         0.03, 0.90, '$Ru2%$ $=$ ${0:.2f}$ $m$'.format(np.float(ru2)),
    #         transform = ax.transAxes,
    #         fontsize = 12,
    #         bbox = props,
    #     )

    # def hist_eta(self, ax, xds_table, x, num_bins):
    #     surface= xds_table.isel(Xp=x).Watlev.values
    #     surface = surface[np.isnan(surface) == False]

    #     # the histogram of the data
    #     n, bins, patches = ax.hist(
    #         surface, num_bins,
    #         color = 'khaki',
    #         alpha = 1,
    #         rwidth = 0.92,
    #         density = 1,
    #         zorder = 2,
    #     )

    #     (mu, sigma) = norm.fit(surface)

    #     bins = np.linspace(-(np.nanmax(bins)+1.5), np.nanmax(bins)+1.5, 100)
    #     y = norm.pdf(bins, mu, sigma)
    #     #y = mlab.normpdf(bins, mu, sigma)

    #     ax.plot(
    #         bins, y, '-',
    #         c = 'k',
    #         label = 'Normal distribution',
    #     )

    #     ax.set_xlabel('$Normalized$ $surface$ $elevation$ $(m)$')
    #     ax.set_ylabel('$Probability$ $density$')
    #     ax.set_title('Surface elevation distribution X = WG{0}'.format(x), fontweight='bold')

    #     #ax.set_xlim(bins[0]-0.5, bins[-1]+0.5)
    #     ax.legend(loc='upper right')

    # def POT(self, xds_table, ws, q):
    #     time = xds_table.Tsec.values

    #     x, yQ = [], []
    #     umbral = 0.01  # Minimun value to consider the overtopping
    #     volume = 1      # Initialized

    #     while volume > umbral:

    #         nonq = np.where(q <= umbral)[0]
    #         xp = np.where(q == np.nanmax(q))[0]

    #         pa = nonq[np.where(nonq <= xp)[0][-1]]
    #         ps = nonq[np.where(nonq >= xp)[0][0]]

    #         volume = np.trapz(q[pa:ps], time[pa:ps]) 

    #         if volume != 0:
    #             yQ.append(volume)
    #             x.append(time[np.int(xp)])

    #             qt = [i for i in range(len(q)) if i not in np.arange(pa, ps, 1)]

    #             q = [q[i] for i in qt]
    #             time = [time[i] for i in qt]

    #             time = np.array(time)
    #             q = np.array(q)

    #     return(x, yQ)

    # def hist_Q(self, ax, ax1, ax2, xds_table, q, x, xQ, yQ):
    #     tendc = self.proj.tendc
    #     time = xds_table.Tsec.values

    #     # the histogram of the data
    #     ax1.plot(q, c='r', label='q (m3/s/m)')

    #     ax2.scatter(
    #         xQ[0], yQ[0],
    #         s = 3,
    #         c = 'b',
    #         label = 'Q event (m3/m)',
    #     )
    #     ax2.set_xlabel('Time (s)')

    #     for qi in range(len(xQ)):
    #         ax2.scatter(xQ[qi], yQ[qi], s = 6, c = 'b')
    #         ax2.plot(
    #             [xQ[qi], xQ[qi]], [0, yQ[qi]],
    #             c = 'cornflowerblue',
    #             linewidth = 1,
    #         )

    #     num_bins = 15
    #     n, bins, patches = ax.hist(
    #         q, num_bins,
    #         color = 'coral',
    #         rwidth = 0.50,
    #         zorder = 2,
    #         density = 1,
    #     )
    #     alpha, loc, scale = pareto.fit(q, floc=q.min(), fscale=q.std())

    #     x = np.linspace(0, bins[-1], len(q))
    #     ax.plot(x, pareto.pdf(x, b=alpha, scale=scale), c='k', label='Pareto distribution')

    #     # Fit Pareto distributioin PDF
    #     mean, var, skew, kurt = pareto.stats(q, moments='mvsk')

    #     ax.legend(loc='upper right')
    #     ax1.legend(loc='upper right')
    #     ax2.legend(loc='upper right')

    #     ax.set_xlim(bins[0], bins[-1])
    #     ax1.set_xlim(0, tendc)
    #     ax2.set_xlim(0, tendc)

    #     ax1.set_ylim(0, np.nanmax(q)+0.05)
    #     ax2.set_ylim(0, np.nanmax(yQ)+0.05)

    #     ax.set_xlabel('$Normalized$ $Overtopping$ $(m3/m)$')
    #     ax.set_ylabel('$Probability$ $density$')
    #     ax.set_title('Overtopping distribution', fontweight='bold')

    # def limits_hi(self, ax, xds_table, ws, ds_fft_hi, nbins):
    #     gate = ws['Gate_Q'].values
    #     maxh = 0
    #     minh = 100
    #     limx=-100

    #     for wg in range(gate[0]-30):
    #         if wg in ds_fft_hi.Xp.values: # Dry-wet boundary
    #             Hi = ds_fft_hi.sel(Xp=wg).Hi.values
    #             Hi = Hi[np.isnan(Hi) == False]
    #             y, x = np.histogram(Hi, bins=nbins, density=1)
    #             if np.nanmax(y) > maxh: maxh = np.nanmax(y)
    #             if np.nanmin(y) < minh: minh = np.nanmin(y)
    #             if np.nanmax(Hi) > limx: limx = np.nanmax(Hi)

    #     #ax.set_ylim(minh, maxh)
    #     ax.set_xlim(0, limx)

    # def limits_eta(self, ax, xds_table, ws, nbins):
    #     gate = ws['Gate_Q'].values
    #     maxeta = -100
    #     mineta = 100

    #     limx = -100

    #     for wg in range(gate[0]-30):
    #         if wg in xds_table.Xp.values: # Dry-wet boundary
    #             eta = xds_table.sel(Xp=wg).Watlev.values
    #             eta = eta[np.isnan(eta) == False]
    #             y, x = np.histogram(eta, bins=nbins, density=1)

    #             if np.nanmax(y) > maxeta: maxeta = np.nanmax(y)
    #             if np.nanmin(y) < mineta: mineta = np.nanmin(y)
    #             if np.nanmax(eta) > limx: limx = np.nanmax(eta)

    #     ax.set_ylim(mineta, maxeta)
    #     ax.set_xlim(-limx, limx)

    # def video_summary_output(self, ws, post, p_run, depth,
    #                          step_video=1, clip_duration=1):
    #     '''
    #     Figure-frames for a X-dependent-video with:
    #         Hs, Hrms, Hmax
    #         evolution of free surface
    #         evolution of spectrum

    #     ws  -        waves-dataset iterrow
    #     xds_table -  Dataset with output vars
    #     df -         setup dataframe
    #     df_Hi -      FFt output
    #     p_run -      output file path
    #     depth -      bathymetry values
    #     runrup -     bool var
    #     '''

    #     # data from postprocessed output
    #     xds_table = post['table_out']
    #     gate = post['Gate_Q']
    #     df = post['su']
    #     df_Hi = post['df_spec']

    #     # data from wave state
    #     H = ws['H']
    #     T = ws['T']
    #     WL = ws['WL']

    #     # init figure
    #     fig = plt.figure(figsize=(20, 9))
    #     gs = GridSpec(4, 10)
    #     gs.update(wspace=1.1,hspace=0.30)

    #     plt.ioff()  # mode off interactive graphics

    #     # video frames path
    #     p_frames = op.join(p_run, 'video_so_frames')
    #     if os.path.exists(p_frames):
    #         shutil.rmtree(p_frames)
    #     os.mkdir(p_frames)

    #     ax0 = fig.add_subplot(gs[0, :5])     # Hsi contribution
    #     ax1 = fig.add_subplot(gs[1, :5])     # HsIc, HsIg, HsVlf Validation
    #     ax2 = fig.add_subplot(gs[2:4, :5])   # Shore-cross profile
    #     ax3 = fig.add_subplot(gs[-1, -5:-2]) # Watlev series
    #     ax4 = fig.add_subplot(gs[-1, -2:])   # Spectrum
    #     ax5 = fig.add_subplot(gs[0, -5:])    # Forcing
    #     ax6 = fig.add_subplot(gs[1, -5:])    # Runup
    #     ax7 = fig.add_subplot(gs[2, -5:])    # Overtopping discharge

    #     # Save figures to create a video
    #     for figure, wg in enumerate(np.arange(0, int(gate), 5)):

    #         x = np.where(xds_table.Xp.values >= wg)[0][0]
    #         self.output_contribution(ax0, df_Hi)
    #         self.output_fft(ax1, df_Hi)
    #         self.output_env(ax2, ws, post, xds_table, df, df_Hi, depth)
    #         self.output_gate_series(ax3, ws, xds_table, x)
    #         self.output_gate_spectrum(ax4, ws, xds_table, x)
    #         self.output_forcing(ax5, 0, xds_table, 0, 0)
    #         self.output_gate_runup(ax6, post, xds_table, 0, 0)
    #         self.output_gate_disch(ax7, post, xds_table, 0, 0)

    #         # plot in ax1 the WG indicator
    #         ax2.scatter(wg, 0, s = 15, c = 'b', zorder = 2)
    #         ax2.plot(
    #             [wg, wg, wg, wg, wg], [0, 0.2, 0.3, 0.4, 0.5],
    #             color = 'deepskyblue',
    #             zorder = 1,
    #         )
    #         ax2.annotate('WG {0}'.format(wg), xy=(wg-4, 0.90), rotation = 'vertical')

    #         #if ws.forcing.values == 'Jonswap':
    #         if 'gamma' in ws.keys():
    #             ax0.set_title(
    #                 'Jonswap $H_s={0}$ $m$ $T_p={1:.2f}$ $s$ $\u03B3$ $=$ ${2}$ $WL={3}$ $m$'.format(
    #                     np.float(H), np.float(T), ws['gamma'], np.float(WL),
    #                 ),
    #                 fontweight = 'bold',
    #             )
    #         else:
    #             ax0.set_title(
    #                 '$H={0}$ $m$ $T={1:.2f}$ $s$ $WL={3}$ $m$'.format(
    #                     np.float(H), np.float(T), np.float(WL)
    #                 ),
    #                 fontweight = 'bold',)
    #         path = op.join(p_run, 'X')

    #         # save figure
    #         p_fig = op.join(p_frames, "{0}.png".format(str(figure).zfill(6)))
    #         fig.savefig(p_fig, facecolor='w', pad_inches = 0)

    #         # clear axis
    #         ax0.clear()
    #         ax1.clear()
    #         ax2.clear()
    #         ax3.clear()
    #         ax4.clear()
    #         ax5.clear()
    #         ax6.clear()
    #         ax7.clear()

    #     plt.close(fig)

    #     # use moviepy to generate a video
    #     p_video = op.join(p_run, 'video_so.mp4')
    #     if op.isfile(p_video): os.remove(p_video)

    #     files_sorted = sorted(glob.glob(op.join(p_frames, '*.png')))

    #     clips = [ImageClip(m).set_duration(clip_duration) for m in files_sorted]
    #     concat_clip = concatenate_videoclips(clips, method='compose')
    #     concat_clip.write_videofile(p_video, fps=60, logger=None)

    #     return p_video

    # def video_waves_propagation(self, ws, post, p_run, depth,
    #                             step_video=1, clip_duration=1):
    #     '''
    #     Figure-frames for a Time-dependent-video with:
    #         Boundary forcing
    #         Waves breaking over profile
    #         Run-up excursion
    #         Discharge magnitude /m
    #     '''

    #     # data from postprocessed output
    #     xds_table = post['table_out']

    #     # data from wave state
    #     H = ws['H']
    #     T = ws['T']

    #     # data from grid
    #     dx = self.proj.b_grid.dx
    #     cut_X = np.argmax(np.abs(depth)) * dx

    #     # init figure
    #     fig = plt.figure(figsize=(13, 9))
    #     gs = GridSpec(3, 1)
    #     ax0 = fig.add_subplot(gs[0, :])    # Forcing
    #     ax1 = fig.add_subplot(gs[1:3, :])  # Profile wave-breaking

    #     # video frames path
    #     p_frames = op.join(p_run, 'video_wp_frames')
    #     if os.path.exists(p_frames):
    #         shutil.rmtree(p_frames)
    #     os.mkdir(p_frames)

    #     plt.ioff()  # mode off interactive graphics

    #     # save figures to create a video
    #     t_video = len(xds_table.Tsec.values[:])
    #     for figure, p in enumerate(np.arange(0, t_video, step_video)):

    #         t = xds_table.Tsec.values[p]
    #         self.output_forcing(ax0, cut_X, xds_table, p, t)
    #         self.output_profile(ax1, post, xds_table, t, depth)

    #         ax0.set_title(
    #             'Jonswap $H_s={0}$ m $T_p={1:.2f}$ s $\u03B3 = 10$ Time = {2} s'.format(
    #                 np.float(H), np.float(T), int(xds_table.Tsec.values[p])
    #             ),
    #             fontweight = 'bold',
    #         )

    #         # save figure
    #         p_fig = op.join(p_frames, "{0}.png".format(str(figure).zfill(6)))
    #         fig.savefig(p_fig, facecolor='w', pad_inches = 0)

    #         # clear axis
    #         ax0.clear()
    #         ax1.clear()

    #     plt.close(fig)

    #     # use moviepy to generate a video
    #     p_video = op.join(p_run, 'video_wp.mp4')
    #     if op.isfile(p_video): os.remove(p_video)

    #     files_sorted = sorted(glob.glob(op.join(p_frames, '*.png')))

    #     clips = [ImageClip(m).set_duration(clip_duration) for m in files_sorted]
    #     concat_clip = concatenate_videoclips(clips, method='compose')
    #     concat_clip.write_videofile(p_video, fps=60, logger=None)

    #     return p_video

    # def attr_axes(self, ax):
    #     ax.grid(
    #         color = 'lightgrey',
    #         axis = 'both',
    #         which = 'major',
    #         zorder = 1,
    #     )

    # def histograms(self, ws, post, xds_table, g, q, df, df_Hi, ds_fft_hi, depth, p_run):
    #     '''
    #     Figure histogram evolution across the reef profile (Ru, Q, Hi)
    #     '''
    #     gate = post['Gate_Q']
    #     ru2 = post['ru2']

    #     H = ws['H']
    #     T = ws['T']
    #     WL = ws['WL']
    #     nbins = 15

    #     fig = plt.figure(figsize=(15, 15), constrained_layout=True)
    #     gs = GridSpec(6, 2, hspace=0.6)

    #     ax0 = fig.add_subplot(gs[:2, :])    # Profile
    #     ax1 = fig.add_subplot(gs[2:4, 0])    # Hi histogram
    #     ax2 = fig.add_subplot(gs[4:6, 0])    # Ru-Q Histogram
    #     ax3 = fig.add_subplot(gs[2:4, 1])    # Eta
    #     ax4 = fig.add_subplot(gs[4, 1])    # Ru-Q histogram
    #     ax5 = fig.add_subplot(gs[5, 1])      # Ru-Q histogram

    #     # save figures to create a video
    #     for figure, wg in enumerate(np.arange(0, gate-10, 5)):

    #         self.attr_axes(ax1)
    #         self.attr_axes(ax2)
    #         self.attr_axes(ax3)
    #         self.attr_axes(ax4)

    #         x = np.where(xds_table.Xp.values >= wg)[0][0]

    #         if x in ds_fft_hi.Xp.values:
    #             self.hist_Hi(ax1, df_Hi, ds_fft_hi, wg, x, nbins)
    #             self.hist_eta(ax3, xds_table, x, nbins)

    #             if np.all(q==0) == True:
    #                 self.hist_Ru(ax2, ax4, ax5, ru2, g, nbins)
    #             else:
    #                 xQ, yQ = self.POT(xds_table, ws, q)
    #                 self.hist_Q(ax2, ax4, ax5, xds_table, q, x, xQ, yQ)

    #             self.output_env(ax0, ws, xds_table, df, df_Hi, depth)

    #             # keep constant limits
    #             #self.limits_hi(ax1, xds_table, ws, ds_fft_hi, nbins)
    #             #self.limits_eta(ax3, xds_table, ws, nbins)

    #         ax0.set_title(
    #             'Jonswap $H_s={0}$ m $T_p={1:.2f}$ s $\u03B3 = 10$ $WL$ = ${2}$ $m$'.format(
    #                 np.float(H), np.float(T), np.float(WL)
    #             ),
    #             fontweight = 'bold',
    #         )
    #         path = op.join(p_run, 'Hist')

    #         # plot in ax1 the WG indicator
    #         ax0.scatter(wg, 0, s = 15, c = 'b', zorder = 2)
    #         ax0.plot([wg, wg], [WL, 0+0.3], color = 'k', zorder = 1)
    #         ax0.annotate(
    #             'WG {0}'.format(wg),
    #             xy = (wg, 1.7),
    #             zorder = 3,
    #             color = 'k',
    #             rotation = 'vertical',
    #         )

    #         # save figure
    #         if not os.path.exists(path): os.mkdir(path)
    #         fig.savefig(op.join(path, "{0}.png".format(str(figure).zfill(3))), facecolor='w', pad_inches = 0)

    #         # clear axis
    #         #ax0.clear()
    #         #ax1.clear()
    #         #ax2.clear()
    #         #ax3.clear()
    #         #ax4.clear()
    #         #ax5.clear()

    #         plt.ioff()  # mode off interactive graphics
    #     #plt.close(fig)

    #     return fig

