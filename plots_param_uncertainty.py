import os
import sys
import pandas as pd
from pickle import dump, load
import numpy as np
import matplotlib.pyplot as plt
from information_metrics import *
from scipy.stats import spearmanr


sel = 'revision_0'
res_dir = '../../DATA/EEGS/fitted_models_%s' % sel
fig_dir = '../../PROJECTS/Stomatal_conductance_eval/NP_figures'

sites_params = pd.read_csv( '../../DATA/EEGS/sel2_sites_params.csv')
pft_params = pd.read_csv( '../../DATA/EEGS/selected_pft_params.csv')

pft_list = ['NF', 'BF', 'G-C3', 'C-C3', ]
sites = sites_params['site_id'].values

model_colors = ['#04ABC2', '#B7CE63', '#FDE10D']
mod_list = [ 'WUE_d',  'CM_d4', 'SOX_d3']
mod_list_labels = ['WUE',   'CM',  'SOX']


def fig_box(fig_name):
    fig_name = os.path.join(fig_dir, fig_name)
    fig = plt.figure(figsize=(3.75, 3.5))
    whis = [5, 95]

    param_list_0 = ['lww', 'b', 'b1', 'b2', 'kmax', 'a', 'psi_50']
    param_list_labels = [r'$\lambda_{ww}$', r'$\beta_0$', 
                         r"$b_1'$", r"$b_2'$",
                         r'$K_{max}$', r'$a$', '$\psi_{50}$']
    min_params = [10**-6, 0.01, 10**-6, 10**-5, 10**-6, 0.01, -10]
    max_params = [0.005, 5,  0.005, 0.005, 0.005, 10, -0.01]
            
    param_list = []
    data_cv = []
    data_H = []

    for mod, mod_n in zip(mod_list, mod_list_labels):
        print('')
        print('%s\tparam\tCV\tH' % mod_n)
        for p, pn, pmin, pmax in zip(param_list_0, param_list_labels, min_params, max_params):
            x_cv = np.array([])
            x_H = np.array([])
            for site_i in sites:
                pickle_file = [os.path.join(res_dir, fi)
                                    for fi in os.listdir(res_dir) 
                                    if fi.endswith('pickle') and (site_i in fi)][0]
                with open(pickle_file, 'rb') as fp:
                    df = load(fp)
                
                xxx = df[mod][p]
                xx = np.abs(np.array(xxx) - np.nanmean(xxx)) / np.abs(np.nanmean(xxx))
                x_cv = np.concatenate((x_cv, xx), axis=None)

                if p in ['lww', 'b1', 'b2', 'kmax']:
                    bins = np.arange(pmin, pmax, 0.05/1000.)
                elif p =='b':
                    bins = np.arange(pmin, pmax, 0.05)
                else:
                    bins = np.arange(pmin, pmax, 0.1)
                x_H = np.concatenate((x_H, shannon_entropy(xxx, [bins])), axis=None)
                    
            if np.isnan(np.nanmedian(df[mod][p])) == 0:
                param_list.append(pn)
                data_cv.append(x_cv)
                data_H.append(x_H)
                print('\t%s\t%-5.2f\t%-5.1f' % (p, np.nanmedian(x_cv), np.median(x_H)))

    positions = range(1, len(param_list) + 1)
    ax = fig.add_subplot(2, 1, 1)
    for i, j, c in [[0, 2, 0],  [2, 6, 1], [6, len(data_cv)+1, 2]]:
        ax.boxplot(data_cv[i:j],
                    positions=positions[i:j], whis=whis,
                    showfliers=False, widths=0.5, patch_artist=True, 
                    boxprops=dict(facecolor=model_colors[c], color=model_colors[c]),
                    medianprops=dict(color='w', lw=2), flierprops=dict(markeredgecolor=model_colors[c]),
                    capprops=dict(color=model_colors[c], lw=2), whiskerprops=dict(color=model_colors[c], lw=2))

    ax.set_xticks(positions)
    ax.set_xticklabels([])
    ax.set_ylim([0, 1.5])
    ax.set_yticks([0, 0.5, 1, 1.5])
    ax.set_yticklabels(['0', '0.5', '1', '1.5'], fontsize=8)
    ax.set_xlim([0.25, len(param_list) + 0.5])
    ax.set_ylabel(r'(a)  $\frac{\mid X-\overline{X} \mid}{\overline{X}}$', fontsize=10)

    for x in [2.5, 6.5]:
        ax.axvline(x, color='k', linestyle='-', lw=0.75)
    ax.tick_params(direction='inout')
    ax.get_yaxis().set_label_coords(-0.08, 0.5)

    ax = fig.add_subplot(2, 1, 2)
    for i, j, c in [[0, 2, 0],  [2, 6, 1], [6, len(data_cv)+1, 2]]:
        ax.boxplot(data_H[i:j],
                    positions=positions[i:j], whis=whis,
                    showfliers=False, widths=0.5, patch_artist=True, 
                    boxprops=dict(facecolor=model_colors[c], color=model_colors[c]),
                    medianprops=dict(color='w', lw=2), flierprops=dict(markeredgecolor=model_colors[c]),
                    capprops=dict(color=model_colors[c], lw=2), whiskerprops=dict(color=model_colors[c], lw=2))
                                                                    
    ax.set_xticks(positions)
    ax.set_xticklabels(param_list, fontsize=9)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_yticklabels([0, 2, 4, 6], fontsize=8)
    ax.set_ylim([0, 6])
    ax.set_xlim([0.25, len(param_list) + 0.5])
    ax.set_ylabel(r'(b)  H($X$)', fontsize=10)

    for ii, pi in enumerate([0.95, 4.25, 7.5]):
        ax.text(pi, -2.2, mod_list_labels[ii], fontsize=12)

    for x in [2.5, 6.5]:
        ax.axvline(x, color='k', linestyle='-', lw=0.75)
    ax.tick_params(direction='inout')
    ax.get_yaxis().set_label_coords(-0.1, 0.5)

    plt.tight_layout()
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.eps' % fig_name)


def fig_pftviolins(fig_name):
    fig_name = os.path.join(fig_dir, fig_name) 
    fig = plt.figure(figsize=(3.5, 5.75))

    mod_p_list = [['WUE_d', 'WUE', 'lww', r"$\lambda_{ww}$; $b_1'$", 1,  model_colors[0], [0, 5], [0, 2, 4], -0.35], 
                  ['WUE_d', 'WUE', 'b', r"$\beta_0$; $b_2'$", 2,  model_colors[0], [0, 5], [0, 2, 4], -0.35],
                  
                  ['CM_d4', 'CM', 'b1', r"$\lambda_{ww}$; $b_1'$", 1,  model_colors[1], [0, 5], [0, 2, 4], 0.35], 
                  ['CM_d4', 'CM', 'b2',  r"$\beta_0$; $b_2'$", 2,  model_colors[1], [0, 5], [0, 2, 4], 0.35], 
                  ['CM_d4', 'CM','a',  r'$a$', 4,  model_colors[1], [0, 10], [0, 4, 8], -0.35], 
                  ['CM_d4', 'CM','psi_50',  r'$-\psi_{50}$', 5,  model_colors[1], [0, 10], [0, 4, 8], -0.35],
                  
                  ['SOX_d3', 'SOX', 'kmax', r'$K_{max}$', 3, model_colors[2], [0, 5], [0, 2, 4], 0], 
                  ['SOX_d3', 'SOX', 'a',  r'$a$', 4, model_colors[2], [0, 10], [0, 4, 8], 0.35], 
                  ['SOX_d3', 'SOX', 'psi_50',  r'$-\psi_{50}$', 5, model_colors[2], [0, 10], [0, 4, 8], 0.35]]

    print('Model\tParam\tN\tB\tG\tC')
    for mod, mod_label,p,\
        y_label, xi, ci,\
        ylim, yticks, offset in mod_p_list:
        l = '%s\t%s'% (mod_label, p)
        ax = fig.add_subplot(5, 1, xi)
        data = []        
        for pft in pft_list:
            x = np.array([])
            sites_pft = sites_params[(sites_params['pft']==pft)]['site_id'].values
            for site_i in sites_pft:
                pickle_file = [os.path.join(res_dir, fi)
                                for fi in os.listdir(res_dir) 
                                if fi.endswith('pickle') and (site_i in fi)][0]

                with open(pickle_file, 'rb') as pf:
                    df = load(pf)

                xx = np.array(df[mod][p])
                if p in ['lww', 'b1', 'b2', 'kmax']:
                    xx = xx * 1000
                elif p == 'psi_50':
                    xx = -xx
                x = np.concatenate((x, xx), axis=None)

            data.append(x)
            l = l + '\t%-5.2f'%np.median(x)
        print(l)
        positions = np.array([1, 3, 5, 7]) + offset
        violin_parts = ax.violinplot(data, positions=positions, showextrema=False, showmedians=True)
        for pc in violin_parts['bodies']:
            pc.set_facecolor(ci)
            pc.set_alpha(1)        
        violin_parts['cmedians'].set_color('w')
        violin_parts['cmedians'].set_linewidth(2)
        
        #Litterature values
        #if p == 'psi_50':
        #    ax.scatter([1, 3, 5, 7], [4.2, 2.7, 3.1, 3.1],
        #                edgecolor='k', facecolor='None', marker='o')
        #if p == 'a':
        #    ax.scatter([1, 3, 5, 7], [8.7, 5.5, 2.2, 2.2],
        #                edgecolor='k', facecolor='None', marker='o')
        
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xlim([0, 8])
        ax.set_xticks(np.array([1, 3, 5, 7]))

        if p == 'psi_50':
            ax.set_xticklabels(['Needleleaf', 'Broadleaf', 'Grass', 'Crop'], fontsize=10, rotation=15)
        else:
            ax.set_xticklabels([])
        ax.tick_params(direction='inout')
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        ax.get_yaxis().set_label_coords(-0.07, 0.5)

    plt.tight_layout()
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.eps' % fig_name)


def param_corr(): 
    for ii, (mod, p) in enumerate([['WUE_d', ['lww', 'b']],
                                   ['CM_d2', ['b1', 'b2']], 
                                   ['CM_d4', ['b1', 'b2']],
                                   ['CM_d4', ['psi_50', 'a']],
                                   ['CM_d4', ['b1', 'a']],
                                   ['CM_d4', ['b2', 'a']],
                                   ['CM_d4', ['b1', 'psi_50']],
                                   ['CM_d4', ['b2', 'psi_50']],
                                   ['SOX_d3', ['kmax', 'psi_50']], 
                                   ['SOX_d3', ['kmax', 'a']], 
                                   ['SOX_d3', ['psi_50', 'a']]]):
        x = np.array([])
        y = np.array([])
        for site_i in sites:
            pickle_file = [os.path.join(res_dir, fi) for fi in os.listdir(res_dir) 
                                                    if fi.endswith('pickle') and (site_i in fi)][0]
            with open(pickle_file, 'rb') as pf:
                df = load(pf)
            xx = np.array(df[mod][p[0]]) 
            yy = np.array(df[mod][p[1]])
            x = np.concatenate((x, xx/np.median(xx)), axis=None)
            y = np.concatenate((y, yy/np.median(yy)), axis=None)
        print(mod, p, np.round(spearmanr(x, y)[0], 2))



if __name__ == "__main__":
    pass
   
