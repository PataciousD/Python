# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 07:51:33 2022

@author: patrick.davis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
warnings.filterwarnings('ignore')

#import invest returns 
url_csv = 'https://raw.githubusercontent.com/evangelistapaul/MC_EM384/main/invest_portfolio_returns.csv' 
df_market = pd.read_csv(url_csv) 

inv_dict = {'SP500':1, 'DJIA':2, 'REIT':3, 'bond_index':4, 'fixed_income':5}
fit_dict = {'SP500':[], 'DJIA':[], 'REIT':[], 'bond_index':[], 'fixed_income':[]}
parameters_dict = {'SP500':[], 'DJIA':[], 'REIT':[], 'bond_index':[], 'fixed_income':[]}

nper = 360

#Requirement #1: Histogram of the returns;
for key, value in inv_dict.items():
    attr = key
    plt.hist(df_market[attr], bins = 25)
    plt.title("Histogram of " + attr + " Returns")
    plt.show()
    del(attr)

#here are the common continuous distributions that we are likely to need (copy-paste from L33_GOFinPython.html):
list_of_dists = ['alpha','anglit','arcsine','beta','betaprime','bradford','burr','burr12',
                 'cauchy','chi','chi2','cosine', 'dgamma','dweibull','erlang','expon',
                 'exponnorm','exponweib','exponpow','f','fatiguelife','fisk','foldcauchy',
                 'foldnorm','genlogistic','genpareto','gennorm','genexpon','genextreme',
                 'gausshyper','gamma','gengamma','genhalflogistic','gilbrat','gompertz',
                 'gumbel_r','gumbel_l','halfcauchy','halflogistic','halfnorm','halfgennorm',
                 'hypsecant','invgamma','invgauss','invweibull','johnsonsb','johnsonsu',
                 'kstwobign','laplace','levy','levy_l','logistic','loggamma','loglaplace',
                 'lognorm','lomax','maxwell','mielke','nakagami','ncx2','ncf','nct','norm',
                 'pareto','pearson3','powerlaw','powerlognorm','powernorm','rdist','rayleigh',
                 'rice','recipinvgauss','semicircular','t','triang','truncexpon','truncnorm',
                 'tukeylambda','uniform','vonmises','vonmises_line','wald','weibull_min','weibull_max']

#for each of the 5 markets
for key, value in inv_dict.items():
    x = np.array(df_market.iloc[:,value])
    
    fit_results = []
    
    for dist_name in list_of_dists:
        dist = getattr(stats, dist_name)
        fit_parameters = dist.fit(x)
        
        ks_test_results = stats.kstest(x, dist_name, args = fit_parameters)
        fit_results.append((dist_name, ks_test_results[0], ks_test_results[1], key))
        fit_results.sort(key=lambda x:float(x[2]), reverse=True)
    
    fit_dict[key].append(fit_results[0])

##Function to plot ecdf of market returns vs. ecdf of fit-generated rvs
def rv_gen(dist_name, x, var_market):
    ssz = 10000
    ns = []
    
    dist = getattr(stats, dist_name)
    parameters = dist.fit(x)
    
    if len(parameters) == 3:
        a, b, c = parameters
    if len(parameters) == 4:
        a, b, c, d = parameters
        
    if dist_name == 'powernorm':
        ns = stats.powernorm.rvs(a, b, c, size = ssz)
    if dist_name == 'burr12':
        ns = stats.burr12.rvs(a, b, c, d, size = ssz)
    if dist_name == 'mielke':
        ns = stats.mielke.rvs(a, b, c, d, size = ssz)
    if dist_name == 'nakagami':
        ns = stats.nakagami.rvs(a, b, c, size = ssz)
    if dist_name == 'johnsonsu':
        ns = stats.johnsonsu.rvs(a, b, c, d, size = ssz)  
    
    parameters_dict[var_market].append((dist_name, parameters))
    
    ecdf_ns = ECDF(ns)
    ecdf_x = ECDF(x)
    plt.title("ECDF of Actual vs. Generated" + var_market + " " + dist_name)
    plt.plot(ecdf_x.x, ecdf_x.y, 'r', ecdf_ns.x, ecdf_ns.y, '--b')
    plt.show()

#print and plot results of the highest p value model for each market
for key, value in fit_dict.items():
    var_market = key  
    i = inv_dict.get(var_market)
    x = np.array(df_market.iloc[:,i])
    dist_name = fit_dict[var_market][0][0]
    
    print("For {} market annual returns, the {} model is the best fit with KS stat= {} and pvalue= {}"
          .format(var_market, fit_dict[var_market][0][0], fit_dict[var_market][0][1], fit_dict[var_market][0][2]))
    
    rv_gen(dist_name, x, var_market)

## FIT COMPLETE ###############################

#import pay data 
url_csv = 'https://raw.githubusercontent.com/evangelistapaul/MC_EM384/main/Sim_lab_military_pay.csv' 
df_pay = pd.read_csv(url_csv) 

#inflation assumption
inflation = 0.03/12

#create column for pay adjusted with inflation in df_pay
df_pay['pay_i'] = np.zeros(df_pay.shape[0])
for i in range(0, df_pay.shape[0]):
    df_pay['pay_i'].iloc[i] = df_pay['pay'].iloc[i]*((1+inflation)**i)

simulation_results = []

#define an investment strategy
#turn this into a dictionary or a list that can be run as a linear program
dict_allocations = {'sp500':0.2, 'djia':0.15, 'reit':0.15, 'bond':0.3, 'fixed':0.2}

max_equities_allocation = 0.5

#analyze three different investment levels
investment_level = (0.05,0.10,0.15)

for j in (0,1,2):
    
    simulation_count = 500
    for sim in range(0,simulation_count):
        
        df_invest = pd.DataFrame()
        df_invest = df_invest.append(df_pay)
        
        investment_in_sp500 = dict_allocations.get('sp500') * investment_level[j]
        a, b, c = parameters_dict.get('SP500')[0][1]
        df_invest['sp500_rv_return'] = (pd.DataFrame(stats.powernorm.rvs(a, b, c, size = nper))/12) + 1
        df_invest['sp500_inv'] = np.zeros(df_invest.shape[0])
        df_invest['sp500_balance'] = np.zeros(df_invest.shape[0])
        sp500_b0 = 0
        
        investment_in_djia = dict_allocations.get('djia') * investment_level[j]
        a, b, c, d = parameters_dict.get('DJIA')[0][1]
        df_invest['djia_rv_return'] = (pd.DataFrame(stats.mielke.rvs(a, b, c, d, size = nper))/12) + 1
        df_invest['djia_inv'] = np.zeros(df_invest.shape[0])
        df_invest['djia_balance'] = np.zeros(df_invest.shape[0])
        djia_b0 = 0
        
        investment_in_reit = dict_allocations.get('reit') * investment_level[j]
        a, b, c, d = parameters_dict.get('REIT')[0][1]
        df_invest['reit_rv_return'] = (pd.DataFrame(stats.johnsonsu.rvs(a, b, c, d, size = nper))/12) + 1 
        df_invest['reit_inv'] = np.zeros(df_invest.shape[0])
        df_invest['reit_balance'] = np.zeros(df_invest.shape[0])
        reit_b0 = 0
        
        investment_in_bond = dict_allocations.get('bond') * investment_level[j]
        a, b, c, d = parameters_dict.get('bond_index')[0][1]
        df_invest['bond_rv_return'] = (pd.DataFrame(stats.burr12.rvs(a, b, c, d, size = nper))/12) + 1
        df_invest['bond_inv'] = np.zeros(df_invest.shape[0])
        df_invest['bond_balance'] = np.zeros(df_invest.shape[0])
        bond_b0 = 0
        
        investment_in_fixed = dict_allocations.get('fixed') * investment_level[j]
        a, b, c = parameters_dict.get('fixed_income')[0][1]
        df_invest['fixed_rv_return'] = (pd.DataFrame(stats.nakagami.rvs(a, b, c, size = nper))/12) + 1
        df_invest['fixed_inv'] = np.zeros(df_invest.shape[0])
        df_invest['fixed_balance'] = np.zeros(df_invest.shape[0])
        fixed_b0 = 0
        
        df_invest['total_balance'] = np.zeros(df_invest.shape[0])
        df_invest['equities_balance'] = np.zeros(df_invest.shape[0])
        df_invest['non_equities_balance'] = np.zeros(df_invest.shape[0])
        df_invest['equities_portion'] = np.zeros(df_invest.shape[0])
        
        sell_eq_amount = 0
        
        for i in range(0, df_invest.shape[0]):
            df_invest['sp500_inv'][i] = (df_invest['pay_i'][i] * investment_in_sp500) - \
                (sell_eq_amount * (dict_allocations.get('sp500') / (dict_allocations.get('sp500') + dict_allocations.get('djia') + dict_allocations.get('reit'))))
            df_invest['sp500_balance'][i] = (df_invest['sp500_inv'][i] + sp500_b0) * df_invest['sp500_rv_return'][i]
            sp500_b0 = df_invest['sp500_balance'][i] 
            
            df_invest['djia_inv'][i] = df_invest['pay_i'][i] * investment_in_djia - \
                (sell_eq_amount * (dict_allocations.get('djia') / (dict_allocations.get('sp500') + dict_allocations.get('djia') + dict_allocations.get('reit'))))
            df_invest['djia_balance'][i] = (df_invest['djia_inv'][i] + djia_b0) * df_invest['djia_rv_return'][i]
            djia_b0 = df_invest['djia_balance'][i] 
            
            df_invest['reit_inv'][i] = df_invest['pay_i'][i] * investment_in_reit - \
                (sell_eq_amount * (dict_allocations.get('djia') / (dict_allocations.get('sp500') + dict_allocations.get('djia') + dict_allocations.get('reit'))))
            df_invest['reit_balance'][i] = (df_invest['reit_inv'][i] + reit_b0) * df_invest['reit_rv_return'][i]
            reit_b0 = df_invest['reit_balance'][i] 
            
            df_invest['bond_inv'][i] = df_invest['pay_i'][i] * investment_in_bond + \
                (sell_eq_amount * (dict_allocations.get('bond') / (dict_allocations.get('bond') + dict_allocations.get('fixed'))))
            df_invest['bond_balance'][i] = (df_invest['bond_inv'][i] + bond_b0) * df_invest['bond_rv_return'][i]
            bond_b0 = df_invest['bond_balance'][i] 
            
            df_invest['fixed_inv'][i] = df_invest['pay_i'][i] * investment_in_fixed + \
                (sell_eq_amount * (dict_allocations.get('fixed') / (dict_allocations.get('bond') + dict_allocations.get('fixed'))))
            df_invest['fixed_balance'][i] = (df_invest['fixed_inv'][i] + fixed_b0) * df_invest['fixed_rv_return'][i]
            fixed_b0 = df_invest['fixed_balance'][i] 
            
            df_invest['total_balance'][i] = df_invest['sp500_balance'][i] + \
                df_invest['djia_balance'][i] + df_invest['reit_balance'][i] + \
                    df_invest['bond_balance'][i] + df_invest['fixed_balance'][i]
                    
            df_invest['equities_balance'][i] = df_invest['sp500_balance'][i] + \
                df_invest['djia_balance'][i] + df_invest['reit_balance'][i]
                
            df_invest['non_equities_balance'][i] = df_invest['total_balance'][i] - df_invest['equities_balance'][i]
                
            df_invest['equities_portion'][i] = df_invest['equities_balance'][i] / df_invest['total_balance'][i]
            
            if df_invest['equities_portion'][i] > max_equities_allocation:
                #change allocations: a) sell / buy, b) change investment allocations
                sell_eq_amount = df_invest['equities_balance'][i] - df_invest['non_equities_balance'][i]
                
        #take the final balance of each simulation (the 360th) and put it into a list
        simulation_results.append((j,df_invest['total_balance'][359]))

df_sim_results = pd.DataFrame(simulation_results, columns = ['inv_level','sim_result'])

results_5 = []
results_10 = []
results_15 = []

for row in range(0, df_sim_results.shape[0]):
    if df_sim_results['inv_level'][row].item() == 0:
        results_5.append(df_sim_results['sim_result'][row])
    if df_sim_results['inv_level'][row].item() == 1:
        results_10.append(df_sim_results['sim_result'][row])
    if df_sim_results['inv_level'][row].item() == 2:
        results_15.append(df_sim_results['sim_result'][row])                     


x = np.array(results_5)    
ex = ECDF(x)
plt.title("5% ECDF")
plt.plot(ex.x, ex.y, 'r')  
plt.show() 

x = np.array(results_10)    
ex = ECDF(x)
plt.title("10% ECDF")
plt.plot(ex.x, ex.y, 'r')  
plt.show() 

x = np.array(results_15)    
ex = ECDF(x)
plt.title("15% ECDF")
plt.plot(ex.x, ex.y, 'r')  
plt.show()


##stacked bar chart iterated through columns:
df_invest_plot = df_invest[['sp500_balance','djia_balance','bond_balance','fixed_balance','reit_balance']]

fig, ax = plt.subplots()

bottom = np.zeros(len(df_invest_plot))

for i, col in enumerate(df_invest_plot.columns):
    ax.bar(df_invest_plot.index, df_invest_plot[col], bottom = bottom, label = col)
    bottom += np.array(df_invest_plot[col])

ax.set_title('Stacked Bar Chart of Portfolio Value Over 360 Periods')
ax.legend()
