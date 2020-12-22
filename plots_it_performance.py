import os
import sys
import numpy as np
from pickle import dump, load
import matplotlib.pyplot as plt
import pandas as pd
from information_metrics import *
from scipy.stats import spearmanr, pearsonr
from matplotlib.lines import Line2D


sites_params = pd.read_csv( '../../DATA/EEGS/sel2_sites_params.csv')
sites = sites_params['site_id'].values

sel = 'revision_0'
res_dir = '../../DATA/EEGS/fitted_models_%s' % sel
fig_dir = '../../PROJECTS/Stomatal_conductance_eval/NP_figures'

timescale = 'd'
mod_list = ['WUE_%s' % timescale, 'CM_%s4' % timescale, 'SOX_%s3' % timescale]
mod_list_labels = [r'WUE', r'CM', r'SOX']
mod_list_nparams= [2, 4, 3]
model_colors = ['#04ABC2', '#B7CE63', '#FDE10D']

markers = [ 'o', 's', '^']

target_m = 'LE'
target_o = 'LE_F_MDS'
source_1 = 'S'
source_2 = 'VPD_a'
sensitivity_var = source_1
n_bins = 15
whis = [5, 95]


def get_result_dict():
    result = {}
    for l in ['site', 'general']:
        result[l] = {}
        for k in [ 'Wet', 'Dry', 'Mesic', 'Full']:
            result[l][k] = {}
            for a in ['aic', 'a_p', 'a_ft', 'a_fu1', 'a_fu2', 'a_fs', 'a_fr', 'a_fp']: 
                result[l][k][a] = []

    for site_i in sites:
        data_file =  [os.path.join(res_dir, fi) for fi in os.listdir(res_dir) 
                                                        if fi.endswith('gen.csv') and (site_i in fi)][0]
        data_0 = pd.read_csv(data_file, header = 0, index_col = 0, parse_dates = True, 
                          infer_datetime_format = True,)
        
        vv = [source_1, source_2, target_o, ]
        for mi in mod_list:
            vv.append('%s_%s' % (target_m, mi))
            vv.append('%s_%s_gen' % (target_m, mi))
        data_0 = data_0[vv].dropna()

        p25 = np.percentile(data_0[sensitivity_var].values, 25)
        p75 = np.percentile(data_0[sensitivity_var].values, 75)

        data_dry = data_0[data_0[sensitivity_var] <= p25]
        data_med = data_0[(data_0[sensitivity_var] < p75) & (data_0[sensitivity_var] > p25)]
        data_wet = data_0[data_0[sensitivity_var] >= p75]
        
        for tag, dd in zip(['Mesic', 'Wet', 'Dry', 'Full'], [data_med, data_wet, data_dry, data_0]):
            site_crit = []
            for mi, nparams in zip(mod_list, mod_list_nparams):
                target_g = '%s_%s_gen' % (target_m, mi)
                aic = aikaike_criterion_rss(dd[target_o].values, dd[target_g].values, nparams)
                site_crit.append(aic)
            x_s = list(np.sort(site_crit))
            result['general'][tag]['aic'].append([x_s.index(xi)  for xi in site_crit])

        for tag, dd in zip(['Mesic', 'Wet', 'Dry', 'Full'], [data_med, data_wet, data_dry, data_0]):
            site_crit = []
            for mi, nparams in zip(mod_list, mod_list_nparams):
                target = '%s_%s' % (target_m, mi)
                aic = aikaike_criterion_rss(dd[target_o].values, dd[target].values, nparams)
                site_crit.append(aic)
            x_s = list(np.sort(site_crit))
            result['site'][tag]['aic'].append([x_s.index(xi)  for xi in site_crit])

        
        for tag, dd in zip(['Mesic', 'Wet', 'Dry', 'Full'], [data_med, data_wet, data_dry, data_0]):
            site_crit = []
            for mi in mod_list:
                target_g = '%s_%s_gen' % (target_m, mi)
                a = cal_it_performance(dd, target_g, target_o, source_1, source_2, nbins=n_bins, norm=1)
                site_crit.append(a)
            site_crit = list(zip(*site_crit))
            for ai, k in zip(site_crit, ['a_p', 'a_fu1', 'a_fu2', 'a_fs', 'a_fr', 'a_ft', 'a_fp']):
                result['general'][tag][k].append(ai)

        for tag, dd in zip(['Mesic', 'Wet', 'Dry' ,'Full'], [data_med, data_wet, data_dry, data_0]):
            site_crit = []
            for mi in mod_list:
                target = '%s_%s' % (target_m, mi)
                a = cal_it_performance(dd, target, target_o, source_1, source_2, nbins=n_bins, norm=1)
                site_crit.append(a)
            site_crit = list(zip(*site_crit))
            for ai, k in zip(site_crit, ['a_p', 'a_fu1', 'a_fu2', 'a_fs', 'a_fr', 'a_ft', 'a_fp']):
                result['site'][tag][k].append(ai)
    for l in ['site', 'general']:
        for k in ['Wet', 'Dry', 'Mesic', 'Full']:
            for a in ['aic', 'a_p', 'a_ft', 'a_fu1', 'a_fu2', 'a_fs', 'a_fr', 'a_fp']:
                result[l][k][a] = list(zip(*result[l][k][a]))
    return result


def fig_prediction(fig_name):
    fig_name = os.path.join(fig_dir, fig_name)
    fig = plt.figure(figsize=(5.5, 5.5))
    
    result = get_result_dict()

    positions=range(1, len(mod_list_labels) + 1)
    ax = fig.add_subplot(3, 3, 7)
    ax.axhline(0, linestyle=':', color='k')
    ax = fig.add_subplot(3, 3, 8)
    ax.axhline(0, linestyle=':', color='k')
    ax = fig.add_subplot(3, 3, 9)
    ax.axhline(0, linestyle=':', color='k')
    for spi, (a_i, a_i_name, ymin, ymax) in enumerate([['aic', '(a) AIC rank', 0.5, len(mod_list_labels) + 0.5],
                                                        ['a_p', r'(b) $A_{p}$', 0.4, 0.85],
                                                        ['d_a_p', r'(c) $\Delta$ $A_{p}$', -6, 6],
                                                      ]):

        for cii, condition in enumerate(['Dry', 'Mesic', 'Wet']):
            ax = fig.add_subplot(3, 3, 1 + cii + 3 * spi)
            if a_i == 'aic':
                ax.title.set_text(condition)
            for i, c in enumerate(model_colors):
                if a_i == 'd_a_p':
                    x = (np.array(result['site'][condition]['a_p'][i]) - np.array(result['general'][condition]['a_p'][i])) * 100
                else:
                    x = result['general'][condition][a_i][i]
                if  a_i == 'aic':
                    x = np.array(x) + 1
                    ax.boxplot([x, ], positions=[positions[i], ], showfliers=False, 
                                   whis=whis, widths=0.5, patch_artist=True,
                                   boxprops=dict(facecolor=c, color=c),
                                   medianprops=dict(color='w', lw=0), flierprops=dict(markeredgecolor=c),
                                   capprops=dict(color=c, lw=2), whiskerprops=dict(color=c, lw=2))
                else:
                    ax.boxplot([x, ], positions=[positions[i], ], showfliers=False, 
                                   whis=whis, widths=0.5, patch_artist=True,
                                   boxprops=dict(facecolor=c, color=c),
                                   medianprops=dict(color='w', lw=2), flierprops=dict(markeredgecolor=c),
                                   capprops=dict(color=c, lw=2), whiskerprops=dict(color=c, lw=2))

            
            ax.set_xlim([0.5, len(mod_list_labels) + 0.5])
            if a_i == 'aic':
                ax.set_xticks(positions)
                ax.set_yticks(range(1, len(mod_list_labels) + 1))
                ax.set_xticklabels([])
            elif a_i == 'a_p':
                ax.set_xticks(positions)
                ax.set_xticklabels([])
                ax.set_yticks([0.4, 0.6, 0.8])
            else:
                ax.set_xticks(positions)
                ax.set_yticks([-5,  0, 5])
                ax.set_xticklabels(mod_list_labels, fontsize=10)
            if condition != 'Dry':
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(a_i_name, fontsize=12)
            ax.get_yaxis().set_label_coords(-0.25,0.5)
            ax.tick_params(direction='inout')
            ax.tick_params(direction='inout')
            ax.set_ylim([ymin, ymax])
            
    plt.tight_layout()
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.eps' % fig_name)


def fig_function(fig_name):
    fig_name = os.path.join(fig_dir, fig_name)
    fig = plt.figure(figsize=(5.5, 8))
    
    result = get_result_dict()
    positions=range(1, len(mod_list_labels) + 1)
    iter_criteria = [['a_ft', r'(a) $A_{f, T}$', -0.1, 0.85],
                      ['a_fu1', r'(b) $A_{f, \theta}$', -0.125, 0.125],
                      ['a_fu2', r'(c) $A_{f, D}$', -0.05, 0.35],
                      ['a_fs', r'(d) $A_{f, S}$', -0.35, 0.05],
                      ['a_fr', r'(e) $A_{f, R}$',  -0.007, 0.007],
                      ]
    for spi, (a_i, a_i_name, ymin, ymax) in enumerate(iter_criteria):

        for cii, condition in enumerate(['Dry', 'Mesic', 'Wet']):
            ax = fig.add_subplot(5, 3, 1 + cii + 3 * spi)
            if a_i == 'a_ft':
                ax.title.set_text(condition)

            ax.axhline(0, linestyle=':', color='k')
            for i, c in enumerate(model_colors):
                x = result['general'][condition][a_i][i]
                ax.boxplot([x, ], positions=[positions[i], ], showfliers=False, 
                               whis=whis, widths=0.5, patch_artist=True,
                               boxprops=dict(facecolor=c, color=c),
                               medianprops=dict(color='w', lw=2), flierprops=dict(markeredgecolor=c),
                               capprops=dict(color=c, lw=2), whiskerprops=dict(color=c, lw=2))

            ax.tick_params(direction='inout')
            ax.set_ylim([ymin, ymax])
            ax.set_xlim([0.5, len(mod_list_labels) + 0.5])

            if a_i == 'a_fr':
                ax.set_xticks(positions)
                ax.set_xticklabels(mod_list_labels, fontsize=10)
            else:
                ax.set_xticks(positions)
                ax.set_xticklabels([])
            if condition != 'Dry':
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(a_i_name, fontsize=12)
                ax.get_yaxis().set_label_coords(-0.35,0.5)

            if (a_i == 'a_fr') and (condition == 'Dry'):
                ax.set_yticks([-0.005, 0, 0.005])
                ax.set_yticklabels([-0.005, '0', 0.005])


    plt.tight_layout()
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.eps' % fig_name)


def fig_scatter_A(fig_name):
    fig_name = os.path.join(fig_dir, fig_name)
    fig = plt.figure(figsize=(7.5, 7.5))

    pft_list = ['NF', 'BF', 'G-C3', 'C-C3']
    pft_subix = [3, 4, 7, 8]
    mod_list_ = ['SOX_d3',  'CM_d4', 'WUE_d']
    colors_ = [model_colors[2], model_colors[1], model_colors[0]]
    markers_ = [markers[2], markers[1], markers[0]]

    legend_elements = [Line2D([0], [0], marker= markers[0], linestyle='', color=model_colors[0], label=r'$WUE$'),
                        Line2D([0], [0], marker=markers[1], linestyle='', color=model_colors[1], label=r'$CM$'),
                        Line2D([0], [0], marker=markers[2], linestyle='', color=model_colors[2], label=r'$SOX$')]
    
    for mod, color, marker in zip(mod_list_, colors_, markers_):
        results = []
        for site_i in sites:
            pft = sites_params[(sites_params['site_id'] == site_i)]['pft'].values[0]
            subix = pft_subix[pft_list.index(pft)]
            data_file =  [os.path.join(res_dir, fi) for fi in os.listdir(res_dir) 
                                                        if fi.endswith('gen.csv') and (site_i in fi)][0]
        
            data_0 = pd.read_csv(data_file, header = 0, index_col = 0, parse_dates = True, 
                              infer_datetime_format = True,)

            target_g = 'LE_%s_gen' % mod
            vv = ['LE_F_MDS', 'S', 'VPD_a', target_g ]
            data_0 = data_0[vv].dropna()
            
            p25 = np.percentile(data_0['S'].values, 25)
            p75 = np.percentile(data_0['S'].values, 75)
            data_dry = data_0[data_0['S'] <= p25]
            data_med = data_0[(data_0['S'] < p75) & (data_0['S'] > p25)]
            data_wet = data_0[data_0['S'] >= p75]

            x_fp = []
            x_ft = []
            x_p = []
            for dd in [data_wet, data_dry, data_med]:
                a_p, a_fu1, a_fu2, a_fs, a_fr, a_ft, a_f = cal_it_performance(dd, target_g, 'LE_F_MDS', 'S', 'VPD_a', nbins=15, norm=1)
                x_fp.append(np.abs(a_fu1) + np.abs(a_fu2) + np.abs(a_fs) + np.abs(a_fr))
                x_ft.append(np.abs(a_ft))
                x_p.append(np.abs(a_p))
            x_fp = np.mean(x_fp)
            x_ft = np.mean(x_ft)
            x_p = np.mean(x_p)
            results.append([x_fp, x_ft, x_p])

            ax = fig.add_subplot(2, 2, 1)
            ax.scatter(x_fp, x_p, marker=marker, color=color, s=30)
            ax = fig.add_subplot(2, 2, 3)
            ax.scatter(x_ft, x_p, marker=marker, color=color, s=30)
            ax = fig.add_subplot(2, 2, 4)
            ax.scatter(x_ft, x_fp, marker=marker, color=color, s=30)
            ax = fig.add_subplot(4, 4, subix)
            ax.scatter(x_fp, x_p, marker=marker, color=color, s=20)
        x_fp, x_ft, x_p = list(zip(*results))
        #print (mod[:3], np.round(spearmanr(x_fp, x_p)[0], 2), np.round(spearmanr(x_ft, x_p)[0], 2), np.round(spearmanr(x_fp, x_ft)[0], 2))
    ax = fig.add_subplot(2, 2, 1)
    ax.set_xlim([0, 0.8])
    ax.set_ylim([0.35, 0.85])
    ax.set_xticks([0, 0.3, 0.6])
    ax.set_yticks([0.4, 0.6, 0.8])
    ax.tick_params(direction='inout')
    ax.set_ylabel('$A_{p}$', fontsize=14)
    ax.set_xlabel(r'$A_{f,P}$', fontsize=14)
    plt.legend(handles=legend_elements, frameon=False, ncol=1, fontsize=10, loc='lower right')
    ax.text(0.03, 0.80, '(a)', fontsize=14)

    ax = fig.add_subplot(2, 2, 3)
    ax.set_xlim([0, 0.8])
    ax.set_ylim([0.35, 0.85])
    ax.set_xticks([0, 0.3, 0.6])
    ax.set_yticks([0.4, 0.6, 0.8])
    ax.tick_params(direction='inout')
    ax.set_ylabel('$A_{p}$', fontsize=14)
    ax.set_xlabel('$|A_{f,T}|$', fontsize=14)
    ax.text(0.03, 0.80, '(b)', fontsize=14)

    ax = fig.add_subplot(2, 2, 4)
    ax.set_xlim([0, 0.8])
    ax.set_ylim([0, 0.8])
    ax.set_xticks([0, 0.3, 0.6])
    ax.set_yticks([0, 0.3, 0.6])
    ax.tick_params(direction='inout')
    ax.set_xlabel('$|A_{f,T}|$', fontsize=14)
    ax.set_ylabel(r'$A_{f,P}$', fontsize=14)
    ax.text(0.03, 0.73, '(c)', fontsize=14)
    
    for pft, xi in zip(['Needleleaf', 'Broadleaf', 'Grass', 'Crop'], pft_subix):
        ax = fig.add_subplot(4, 4, xi)
        ax.set_xlim([0, 0.8])
        ax.set_ylim([0.35, 0.85])
        ax.set_xticks([0, 0.3, 0.6])
        ax.set_yticks([0.4, 0.6, 0.8])
        ax.tick_params(direction='inout')
        if (xi == 3) or (xi == 4):
            ax.set_xticklabels([])
        if (xi == 8) or (xi == 4):
            ax.set_yticklabels([])

        ax.plot(0,0, marker='', lw=0, label=pft)
        ax.legend(loc='lower right', fontsize=10, frameon=False)
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.eps' % fig_name)


def delta_cond(fig_name=None):
    result = get_result_dict()
    fig = plt.figure(figsize=(5.5, 10.5))
    positions=range(1, len(mod_list_labels) + 1)
    for spi, (a_i, a_i_name, ymin, ymax) in enumerate([['a_p', r'(a) $\Delta A_{p}$', None, None],
                                                      ['a_ft', r'(b) $\Delta A_{f, T}$', None, None],
                                                      ['a_fp', r'(c) $\Delta A_{f}$', None, None],
                                                      ['a_fu1', r'(d) $\Delta A_{f, \theta}$', None, None],
                                                      ['a_fu2', r'(e) $\Delta A_{f, D}$', None, None],
                                                      ['a_fs', r'(f) $\Delta A_{f, S}$', None, None],
                                                      ['a_fr', r'(g) $\Delta A_{f, R}$', None, None],
                                                      ]):

        for cii, condition in enumerate(['Wet - Dry', 'Mesic - Dry', 'Mesic - Wet']):
            ax = fig.add_subplot(7, 3, 1 + cii + 3 * spi)
            ax.axhline(0, linestyle=':', color='k')
            if a_i == 'a_p':
                ax.title.set_text(condition)
            for i, c in enumerate(model_colors):
                if condition == 'Wet - Dry':
                    x = (np.array(result['general']['Wet'][a_i][i]) - np.array(result['general']['Dry'][a_i][i]))
                elif condition == 'Mesic - Dry':
                    x = (np.array(result['general']['Mesic'][a_i][i]) - np.array(result['general']['Dry'][a_i][i]))
                elif condition == 'Mesic - Wet':
                    x = (np.array(result['general']['Mesic'][a_i][i]) - np.array(result['general']['Wet'][a_i][i]))
                ax.boxplot([x, ], positions=[positions[i], ], showfliers=False, 
                               whis=whis, widths=0.5, patch_artist=True,
                               boxprops=dict(facecolor=c, color=c),
                               medianprops=dict(color='w', lw=2), flierprops=dict(markeredgecolor=c),
                               capprops=dict(color=c, lw=2), whiskerprops=dict(color=c, lw=2))
                print('%s\t%s\t%s\t%-5.1f\t%-5.1f\t%-5.1f\t%-5.1f'%(a_i, mod_list[i], condition, np.mean(x)*100, np.median(x)*100, np.min(x)*100, np.max(x)*100))
            ax.tick_params(direction='inout')
            ax.tick_params(direction='inout')
            ax.set_ylim([ymin, ymax])
            ax.set_xlim([0.5, len(mod_list_labels) + 0.5])
            if a_i == 'a_f':
                ax.set_xticks(positions)
                ax.set_xticklabels(mod_list_labels, fontsize=10)
            else:
                ax.set_xticks(positions)
                ax.set_xticklabels([])
            ax.set_ylabel(a_i_name, fontsize=12)
            ax.get_yaxis().set_label_coords(-0.25,0.5)
        print('')
    if fig_name is not None:
        fig_name = os.path.join(fig_dir, fig_name)     
        plt.tight_layout()
        plt.savefig(fig_name)


if __name__ == "__main__":
    pass