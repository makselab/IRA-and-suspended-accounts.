from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tigramite.pcmci import PCMCI
from tigramite import lpcmci
from tigramite import data_processing as pp

from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.parcorr_wls import ParCorrWLS
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.parcorr_mult import ParCorrMult
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.independence_tests.regressionCI import RegressionCI

from tigramite.models import LinearMediation
from tigramite import plotting as tp

import glob
import numpy as np
import os

import time

gran_path = '/sdf/MatteoPaper/'
section_four_path = gran_path + 'section_four/'

resample_freq = '15min'
type_cnt_names = ['OfficialClient']#AllClient

residuals = 'residuals2'
classes = 'causal2'

def print_residuals(residuals):       
    # print(tsts_resid)
    fig, ax1 = plt.subplots(figsize=(20, 6))
    color = 'tab:red'
    ax1.set_ylabel('Pro T (residuals)', color=color)  # we already handled the x-label with ax1
    # ax1.set_ylim((-600, 600))
    ax1.plot("pro_t", data=residuals, color=color, lw=1)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel("Pro H (residuals)", color=color)  # we already handled the x-label with ax1
    # ax2.set_ylim((-20000, 20000))
    ax2.plot("pro_c", data=residuals, color=color, lw=1)
    ax2.tick_params(axis='y', labelcolor=color)
    #plt.savefig(f"fig/c{cN}-{layer}-layer-resid.pdf", dpi=300)
    plt.show()
    #plt.close()
    
def reproduce_bovet_code(residuals,
                         tau_max,
                         tau_min,
                         pc_alpha,
                         net_type,
                         save_dir,
                         fdr_threshold,
                         print_links = False,
                         print_res = False,
                         plus = False,
                         max_comb = 3,
                         test = None):
    
    if print_res:
        print_residuals(residuals)
    

    #selecting groups to analyze
    temp=residuals[[i for i in residuals.columns if i not in 'tot']]
    
    parameters={}
    
    
    var_names = list(temp.columns)
    
    dataframe = pp.DataFrame(data=temp.to_numpy(),var_names = var_names )
    
    if test=='shuffle_test':
        parameters['cond_ind_test'] = 'pars_corr_shuffle_test'
        parcorr = ParCorr(significance='shuffle_test',sig_samples=50)
        
    elif test=='ParCorrMult':
        parameters['cond_ind_test'] = 'ParCorrMult'
        parcorr = ParCorrMult() 
    
    elif test=='RobustParCorr':
        parameters['cond_ind_test'] = 'RobustParCorr'
        parcorr = RobustParCorr() 
    
    elif test=='ParCorrWLS':
        parameters['cond_ind_test'] = 'ParCorrWLS'
        parcorr = ParCorrWLS() 
    
    elif test=='GPDC':
        parameters['cond_ind_test'] = 'GPDC'
        parcorr = GPDC() 
    
    elif test=='RegressionCI':
        parameters['cond_ind_test'] = 'RegressionCI'
        parcorr = Gsquared(significance='analytic') 
    
    else:
        parameters['cond_ind_test']='pars_corr_analytic'
        parcorr=ParCorr()
 
    
    parameters['tau_max'] = tau_max
    parameters['tau_min'] = tau_min
    parameters['pc_alpha'] = pc_alpha
    parameters['var_names'] = var_names

    pcmci = PCMCI(dataframe=dataframe, 
                  cond_ind_test=parcorr,
                  verbosity=False)
    
    #return pcmci
    if plus:
        pls = 'plus'
        results = pcmci.run_pcmciplus(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha)
    else:
        pls = ''
        
        if not max_comb:
            max_comb = 'no_rest'
            results = pcmci.run_pcmci(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha)#
        
        else:
            
            results = pcmci.run_pcmci(tau_min=tau_min, tau_max=tau_max, pc_alpha=pc_alpha,max_combinations=max_comb)
    
    p_matrix = results['p_matrix']
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=p_matrix, fdr_method='fdr_bh')
    
    parameters['p_matrix'] = p_matrix
    parameters['q_matrix'] = q_matrix
    parameters['val_matrix'] = p_matrix
    parameters['fdr_t'] = fdr_threshold

    graph = pcmci.get_graph_from_pmatrix(q_matrix,tau_min=tau_min, tau_max=tau_max, alpha_level=fdr_threshold)
    parameters['graph'] = graph
    
    #%% print results
    if print_links:
        pcmci.print_significant_links(q_matrix,  
                                      results['val_matrix'],
                                      alpha_level = pc_alpha)
        
    #%% get selected parents and fit linear model
    parent_dict = pcmci.return_parents_dict(graph, results['val_matrix'])
    parameters['parent_dict'] = parent_dict
    
    
    med = LinearMediation(dataframe=dataframe) 

    med.fit_model(all_parents=parent_dict)
    parameters['model.phi'] = med.phi
    parameters['model.psi'] = med.psi
    
    if print_links:
        tau_max = med.psi.shape[0]-1
        for j in range(len(var_names)):
            print('CE --> ', var_names[j])
            for tau in range(1, tau_max+1):
                print('tau = ', tau)
                for i in range(len(var_names)):
                    ICE_ijtau = med.get_ce(i, tau, j)
                    if ICE_ijtau != 0.0:
                        print('CE {i} --> {j} = {I}'.format(i=var_names[i],
                                                      j=var_names[j],
                                                      I=ICE_ijtau))
    if print_links:
        #max CE
        for j in range(len(var_names)):
            print('CE --> ', var_names[j])
            for i in range(len(var_names)):
                ICE_ijmax = np.abs(med.psi[1:, j, i]).max()
                if ICE_ijmax != 0.0:
                    print('CE max {i} --> {j} = {I}'.format(i=var_names[i],
                                                  j=var_names[j],
                                                  I=ICE_ijmax))
                    
    #%% confidence interval with bootstrap

    #compute residuals

    def bootstrapping_ar_model(model, num_bs=200, seed=52):



        T, N = model.dataframe.values[0].shape

        std_data = np.zeros((T,N)) 

        #standardize
        for i in range(N):
            std_data[:,i] = (model.dataframe.values[0][:,i] - model.dataframe.values[0][:,i].mean())/model.dataframe.values[0][:,i].std()

        # initial model coeffs
        phi = model.phi

        tau_max = phi.shape[0] - 1 

        residuals = np.zeros((T-tau_max, N))

        for i in range(T-tau_max):

            model_eval = np.zeros((1,N))
            for tau in range(1, tau_max+1):
                model_eval += np.dot(phi[tau],std_data[i+tau_max-tau])

            residuals[i,:] = std_data[i+tau_max,:] - model_eval

        # generate bootstrap data
        bs_models = []
        ts_indexes = np.arange(residuals.shape[0])
        np.random.seed(seed)
        for _ in range(num_bs):
            bs_residuals = residuals[np.random.choice(ts_indexes, size=T, replace=True),:]
            # bs model
            bs_x = np.zeros((T, N))
            for t in range(0,T):
                if t < tau_max:
                    bs_x[t,:] = bs_residuals[t,:]
                else:
                    model_eval = np.zeros((1,N))
                    for tau in range(1, tau_max+1):
                        model_eval += np.dot(phi[tau],bs_x[t-tau])

                    bs_x[t,:] = model_eval + bs_residuals[t,:]
            #fit bs data
            bs_med = LinearMediation(dataframe=pp.DataFrame(data=bs_x)) #, data_transform=False)
            bs_med.fit_model(all_parents=parent_dict)
            bs_models.append(bs_med)

        return bs_models
    
    #%% bootstap!
    use_bootstrap = True


    if use_bootstrap:    
        num_bs = 200
        bs_models = bootstrapping_ar_model(med, num_bs=num_bs)

        parameters['num_bs'] = 200

    else:
        parameters['num_bs'] = None
    #%% CE with uncertainties


    df_CEmax = pd.DataFrame(columns=var_names, index=var_names)

    df_CEmax_mean = pd.DataFrame(columns=var_names, index=var_names)
    df_CEmax_std = pd.DataFrame(columns=var_names, index=var_names)
    df_CEmax_tau = pd.DataFrame(columns=var_names, index=var_names)

    for j in range(len(var_names)):
        print('CE --> ', var_names[j])
        for i in range(len(var_names)):
            ICE_ijmax = np.abs(med.psi[1:, j, i]).max()
            tau_at_max = np.abs(med.psi[1:, j, i]).argmax()+1
            if use_bootstrap:
                ICE_ijmax_dist = np.array([bs_m.psi[tau_at_max, j, i] for bs_m in bs_models])

            if ICE_ijmax != 0.0:
                if use_bootstrap:
                    print('CE max {i} --> {j} = {I:.5f}, ({m:.5f} +- {s:.5f}), tau={tau}'.format(i=var_names[i],
                                              j=var_names[j],
                                              I=ICE_ijmax,
                                              m=np.abs(ICE_ijmax_dist.mean()),
                                              s=ICE_ijmax_dist.std(),
                                              tau=tau_at_max))
                else:
                    print('CE max {i} --> {j} = {I:.5f}, tau={tau}'.format(i=var_names[i],
                                              j=var_names[j],
                                              I=ICE_ijmax,
                                              tau=tau_at_max))
                df_CEmax.loc[var_names[j], var_names[i]] = ICE_ijmax
                df_CEmax_tau.loc[var_names[j], var_names[i]] = tau_at_max
                if use_bootstrap:
                    df_CEmax_mean.loc[var_names[j], var_names[i]] = np.abs(ICE_ijmax_dist.mean())
                    df_CEmax_std.loc[var_names[j], var_names[i]] = ICE_ijmax_dist.std()
 

    #%% save    

    #save_dir=path+'casual/'
    file_name = os.path.join(save_dir, 'pcmci_{pls}_{cond_ind_test}_pc_alpha{pc_alpha}' +\
                         '_tau_max{tau_max}_'+'_{net_type}_max_comb{max_comb}_fdrh_t{fdr_t}'+ '.pickle')

    file_name = file_name.format(pls=pls,
                                 cond_ind_test=parameters['cond_ind_test'],
                                 pc_alpha=parameters['pc_alpha'],
                                 tau_max=parameters['tau_max'],
                                 net_type=net_type,
                                 max_comb=max_comb,
                                 fdr_t=parameters['fdr_t'])
    
    pd.to_pickle({'parameters' : parameters,
              'df_CEmax' : df_CEmax,
              'df_CEmax_mean' : df_CEmax_mean,
              'df_CEmax_std' : df_CEmax_std,
              'df_CEmax_tau' : df_CEmax_tau,
              }, 
              file_name)
    
    
#tests 'shuffle_test','RobustParCorr','ParCorrMult','ParCorrWLS'
def run_casual_analysis( path_residuals ,
                         net_type,
                         save_dir,
                         test = 'ParCorr',
                         max_comb = 3,
                         pc_alpha = 0.001,
                         fdr_threshold = 0.05,
                         tau_max=18,
                         tau_min=0 ):

    
    residuals = pd.read_pickle(path_residuals)
    
    if 'IvsS' in path_residuals:
        residuals = residuals.drop(columns='S')
    
    
    print('****** max comb : ' + str(max_comb))
    print('   +++++ pc_alpha : ' + str(pc_alpha))
    print('   +++++ fdr_threshold : ' + str(fdr_threshold))
    print('   +++++ test : ' + str(test))
    
    reproduce_bovet_code(residuals,
                     tau_max,
                     tau_min,
                     pc_alpha,
                     net_type,
                     save_dir,  
                     fdr_threshold = fdr_threshold,
                     max_comb = max_comb,
                     plus = False,
                     test = test)  

if __name__ == "__main__":

    #19/05/2023
    t0 = time.time()
    for pc_alpha in [0.05]:
        
        for fdr in [0.001]:
            
            for p in [5]:

                for clnt in type_cnt_names:

                    for gr in ['IvsS']:
                        print(gr,'...')
                        path_residuals = section_four_path  + f"{residuals}/UsersCat_" + \
                                             resample_freq + '_' + clnt + '_' + gr +  f"_residuals_{p}_rp"+'.pickle'

                        net_type = resample_freq + '_' + clnt + '_' + gr +  f"_residuals_{p}"

                        save_dir = section_four_path + f"{classes}/"

                        #print(save_dir)
                        #print(net_type)
                        #print(path_residuals)


                        run_casual_analysis( path_residuals ,
                                             net_type,
                                             save_dir,
                                             test = 'ParCorr',
                                             max_comb = 3,
                                             pc_alpha = pc_alpha,
                                             fdr_threshold = fdr,
                                             tau_max=18,
                                             tau_min=0 )
                        
    print('Time series prepared for tigramite in ', time.time()-t0,' seconds') 
