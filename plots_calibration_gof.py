import os
import sys
import pandas as pd
from pickle import dump, load
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


sel = 'revision_0'
res_dir = '../../DATA/EEGS/fitted_models_%s' % sel
fig_dir = '../../PROJECTS/Stomatal_conductance_eval/NP_figures' 
sites_params = pd.read_csv( '../../DATA/EEGS/sel2_sites_params.csv')
sites = sites_params['site_id'].values

model_colors = ['#04ABC2', '#B7CE63', '#FDE10D']
mod_list = ['WUE_d', 'CM_d4', 'SOX_d3']
mod_list_labels = ['WUE', 'CM', 'SOX']


def fig_boxes(fig_name):
    fig_name = os.path.join(fig_dir, fig_name)
    fig = plt.figure(figsize=(3.75, 3))

    positions = range(1, len(mod_list_labels) + 1)
    whis = [5, 95]

    nse = []
    mape = []
    nse_delta = []
    mape_delta = []

    nse_f = []
    mape_f = []

    for k in mod_list:
        nse_m = np.array([])
        mape_m = np.array([])
        nse_dm = np.array([])
        mape_dm = np.array([])

        nse_fi = []
        mape_fi = []
        for site_i in sites:
            pickle_file = [os.path.join(res_dir, fi) \
                                for fi in os.listdir(res_dir) 
                                    if fi.endswith('pickle') and (site_i in fi)][0]
            with open(pickle_file, 'rb') as pf:
                df = load(pf)

            nse_m = np.concatenate((nse_m, np.array(df[k]['nse_LE_val']) * 100), axis=None)
            mape_m = np.concatenate((mape_m, np.array(df[k]['mape_LE_val']) * 100), axis=None)
            
            nse_dmi = (np.array(df[k]['nse_LE_val']) \
                        - np.array(df[k]['nse_LE_cal'])) \
                        / np.array(df[k]['nse_LE_cal']) * 100
            f_1 = np.sum([1 for nv, nc in zip(df[k]['nse_LE_val'], df[k]['nse_LE_cal']) if nv<nc]) / np.float(len(df[k]['nse_LE_cal'])) * 100.

            mape_dmi = (np.array(df[k]['mape_LE_val'])\
                        - np.array(df[k]['mape_LE_cal'])) \
                        / np.array(df[k]['mape_LE_cal']) * 100
            f_2 = np.sum([1 for mv, mc in zip(df[k]['mape_LE_val'], df[k]['mape_LE_cal']) if mv>mc]) / np.float(len(df[k]['mape_LE_cal'])) * 100.
            
            nse_dm = np.concatenate((nse_dm, nse_dmi), axis=None)
            mape_dm = np.concatenate((mape_dm, mape_dmi), axis=None)
            nse_fi.append(f_1)
            mape_fi.append(f_2)
        
        nse.append(nse_m)
        mape.append(mape_m)
        nse_delta.append(nse_dm)
        mape_delta.append(mape_dm)

        nse_f.append(nse_fi)
        mape_f.append(mape_fi)
    
    print('Median goodness-of-fit of validation subsets, all sites and variants combined')
    print('NSE: %-5.0f  MAPE: %-5.0f'% (np.median(nse), np.median(mape)))
    print()
    print('% of subsets with decreased cal->val gof')
    print('\tNSE\tMAPE')
    for mm, fn, fm, in zip(mod_list_labels, nse_f, mape_f):
        print('%s\t%-5.0f\t%-5.0f'% (mm, np.median(fn), np.median(fm)))

    for subix, data, \
        ylabel, xticks, \
        yticks, ylim in [[1, nse, '(a) NSE', [], [40, 60, 80, 100], [40, 100]], 
                        [2, mape, '(b) MAPE', [], [10, 20, 30], [5, 35]],
                        [3, nse_delta, r'(c) $\Delta$ NSE', mod_list_labels, [-5, 0, 5], [-7.5, 7.5]], 
                        [4, mape_delta, r'(d) $\Delta$ MAPE', mod_list_labels, [-5, 0, 5], [-7.5, 7.5]]]:
        
        ax = fig.add_subplot(2, 2, subix)
        
        for i in range(len(mod_list)):  
            ax.boxplot(data[i:i+1], \
                positions=positions[i:i+1], whis=whis,\
                showfliers=False, widths=0.5, patch_artist=True,
                boxprops=dict(facecolor=model_colors[i], color=model_colors[i]),
                medianprops=dict(color='w', lw=2), flierprops=dict(markeredgecolor=model_colors[i]),
                capprops=dict(color=model_colors[i], lw=2), whiskerprops=dict(color=model_colors[i], lw=2))

        ax.set_xticks(range(1, len(mod_list_labels) + 1))
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks,  fontsize=8)
        ax.set_xticklabels(xticks, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.get_yaxis().set_label_coords(-0.25, 0.5)
        ax.tick_params(direction='inout')
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.eps' % fig_name)


def fig_site_ts(fig_name='Fig_4.png'):
    fig_name = os.path.join(fig_dir, fig_name)
    fig = plt.figure(figsize=(7.5, 5))

    site_i = 'US-Blo'
    target_o = 'LE_F_MDS'
    target_m = 'LE'
    ylabel = '(a)\n\nLatent heat flux (LE)'

    data_file =  [os.path.join(res_dir, fi) for fi in os.listdir(res_dir) 
                                                    if fi.endswith('gen.csv') and (site_i in fi)][0]
    data_0 = pd.read_csv(data_file, header = 0, index_col = 0, parse_dates = True, 
                  infer_datetime_format = True)
    
    #f_dd = '/mnt/m/Original_data/FLUXNET/FLUXNET2015_2020dl/unzipped/FLX_US-Blo_FLUXNET2015_FULLSET_1997-2007_1-4/FLX_US-Blo_FLUXNET2015_FULLSET_DD_1997-2007_1-4.csv'
    f_dd = 'M:/Original_data/FLUXNET/FLUXNET2015_2020dl/unzipped/FLX_US-Blo_FLUXNET2015_FULLSET_1997-2007_1-4/FLX_US-Blo_FLUXNET2015_FULLSET_DD_1997-2007_1-4.csv'
    data_dd =  pd.read_csv(f_dd, 
                      header = 0, index_col = 'TIMESTAMP', parse_dates = True, 
                      infer_datetime_format = True, na_values=-9999)
    
    sel_vars = [target_o, ]
    mod_list_ = [mod_list[2], mod_list[1], mod_list[0]]
    model_colors_ = [model_colors[2], model_colors[1], model_colors[0]]
    for mod, color in zip(mod_list_, model_colors_):
        sel_vars.append('%s_%s' % (target_m, mod))
        sel_vars.append('%s_%s_gen' % (target_m, mod))
        data_0['%s_%s_gen' % (target_m, mod)] = data_0['%s_%s_gen' % (target_m, mod)] 
        data_0['%s_%s' % (target_m, mod)] = data_0['%s_%s' % (target_m, mod)] 
    
    data_0 = data_0[sel_vars]
    data_0 = data_0.reindex(pd.date_range(start=data_0.index.min(),
                    end=data_0.index.max(),
                    freq='30min'))

    data_0['DOY'] = data_0.index.dayofyear
    data_dd['DOY'] = data_dd.index.dayofyear
    ax = fig.add_subplot(2, 1, 1)
    
    ax2 = ax.twinx()
    doy = data_dd.groupby(['DOY'])['DOY'].median()
    s = data_dd.groupby(['DOY'])['SWC_F_MDS_1'].mean()/100.
    ax2.plot(doy, s, color='DarkGrey', lw=1.5)
    ax2.set_ylabel(r'Soil moisture ($\theta$)', color='DarkGrey',)
    ax2.set_ylim([0, 0.4])
    ax2.tick_params(axis='y', colors='DarkGrey')

    for mod, color in zip(mod_list_, model_colors_):
        doy = data_0.groupby(['DOY'])['DOY'].median()
        et = data_0.groupby(['DOY'])['%s_%s_gen' % (target_m, mod)].mean()
        ax.scatter(doy, et, color=color, marker='x')

    for mod, color in zip(mod_list_, model_colors_):
        doy = data_0.groupby(['DOY'])['DOY'].median()
        et = data_0.groupby(['DOY'])['%s_%s' % (target_m, mod)].mean()
        ax.scatter(doy, et, edgecolor=color, color = 'none', lw=1)

    doy = data_0.groupby(['DOY'])['DOY'].median()
    et = data_0.groupby(['DOY'])[target_o].mean()
    ax.scatter(doy, et, marker='s', edgecolor='k', color='none', lw=0.75)
    ax.set_xlim([0, 365])
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Day of year')
    ax.tick_params(direction='inout')

    vmax = 677
    for ii, (mod, color, mod_name) in enumerate(zip(mod_list, model_colors, mod_list_labels)):
        ax = fig.add_subplot(2, 3, 4 +ii)
        ax.scatter(data_0[target_o],  data_0['%s_%s' % (target_m, mod)], edgecolor=color, color = 'none')
        ax.scatter(data_0[target_o],  data_0['%s_%s_gen' % (target_m, mod)], marker='x', color = color)
        ax.set_xlabel('Observed  %s' % target_m)
        ax.set_ylim([0, vmax])
        ax.set_xlim([0, vmax])
        ax.plot([0, vmax], [0, vmax], color='k', lw=0.75)

        data_0 = data_0.dropna()
        slope, intercept, r_value, \
        p_value, std_err = stats.linregress(data_0[target_o].values,  data_0['%s_%s' % (target_m, mod)].values)
        print('%s (o) R^2: %-5.2f  %% bias: %-5.0f' %(mod_name, r_value, intercept/np.mean(data_0[target_o].values)*100))
        ax.plot([0, vmax], [intercept, vmax*slope + intercept], linestyle=":", color='k')
        slope, intercept, r_value, \
        p_value, std_err = stats.linregress(data_0[target_o].values,  data_0['%s_%s_gen' % (target_m, mod)].values)
        print('%s (x) R^2: %-5.2f  %% bias: %-5.0f' %(mod_name, r_value, intercept/np.mean(data_0[target_o].values)*100))
        ax.plot([0, vmax], [intercept, vmax*slope + intercept], linestyle="--", color='k')
        ax.text(40, 590, mod_name)
        if ii == 0:
            ax.set_ylabel('(b)\n\nModeled  %s' % target_m)
        else:
            ax.set_yticklabels([])
        ax.tick_params(direction='inout')
    plt.tight_layout()
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.eps' % fig_name)



if __name__ == "__main__":
    pass