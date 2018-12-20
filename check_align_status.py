# # Check the alignment of the Electromagnetic calorimeter using the track matching of electrons

# In[1]:


import ROOT
import root_numpy
import numpy as np
import pandas as pd
ROOT.TH1.AddDirectory(False)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.colors import LogNorm


# Import the information of each electron that is saved in the ROOT file. The result is converted to pandas DataFrames using root_numpy  

# In[2]:


file_name = "AnalysisResults.root"
tree_name = "electron_informationAllTriggers_"

file = ROOT.TFile(file_name)
file.cd()
electrons = ROOT.gDirectory.Get(tree_name)
br= ['charge', 'pz','pt','eta_track','x_track','y_track','z_track','zvtx','energy', 'x_cluster', 'y_cluster', 'z_cluster', 'eta_cluster','phi_cluster','super_module_number','distance_bad_channel', 'is_in_fid_region', 'n_sigma_electron_TPC']
electrons = root_numpy.tree2array(electrons,branches=br )
#branchs removed: ,'eta_track','phi_track','eta_cluster','phi_cluster',
electrons_nocuts = pd.DataFrame(electrons)
electrons = pd.DataFrame(electrons)
electrons.head()


def apply_cuts(data,
               min_TPC_nsig = -1.0,
               max_TPC_nsig = 3.0, 
               e_over_p_cut_min = 0.9,
               e_over_p_cut_max = 1.2,
               pt_min = 3.0,
               pt_max = 10,
               eta_clus_min = 0,
               eta_clus_max = 0.7):
    #pT cut applied before all the cuts, since it reduces the calculation of the other quantities
    data['p'] = np.sqrt(data['pt']**2+data['pz']**2)
    data['e_over_p'] = data['energy']/data['p']

    is_pt = (data['pt'] >= pt_min) & (data['pt'] <= pt_max)
    data = data[is_pt]

    #E/p cut
    is_e_over_p = (data['e_over_p'] >= e_over_p_cut_min) & (data['e_over_p'] <= e_over_p_cut_max)

    #eta_clus_cut
    is_eta_cut = (np.absolute(data['eta_cluster'])> eta_clus_min) & (np.absolute(data['eta_cluster']) < eta_clus_max)

    #TPC PID cut
    is_TPC_PID = (data['n_sigma_electron_TPC'] >= min_TPC_nsig) & (data['n_sigma_electron_TPC'] <= max_TPC_nsig)
    
    #apply all the cuts
    data = data[is_TPC_PID & is_e_over_p  & is_eta_cut]


apply_cuts(electrons)


def Eta(x,y,z):
    cos_theta = z/np.sqrt(x**2 + y ** 2 + z ** 2)
    eta = -0.5* np.log((1.0-cos_theta)/(1.0+cos_theta) )
    return eta

#Calculate phi and eta at the emcal of the track
electrons['phi_track_emc'] = np.arctan2(electrons['y_track'],electrons['x_track'])
electrons['eta_track_emc'] = Eta(electrons['x_track'], electrons['y_track'], electrons['z_track'])
electrons['x_res'] = electrons['x_track'] - electrons['x_cluster']
electrons['y_res'] = electrons['y_track'] - electrons['y_cluster']
electrons['z_res'] = electrons['z_track'] - electrons['z_cluster']
electrons['phi_res'] = electrons['phi_track_emc'] - electrons['phi_cluster']
electrons['eta_res'] = electrons['eta_track_emc'] - electrons['eta_cluster']


#cut bad matches
electrons = electrons[np.abs(electrons['phi_res'])<0.1]
electrons = electrons[np.abs(electrons['eta_res'])<0.1]

fig_test,ax_test= plt.subplots()
counts, xedges,  im  = ax_test.hist(electrons['e_over_p'])
ax_test.set_ylabel('Counts')
ax_test.set_xlabel('E/p')
ax_test.set_yscale('log')


def plot_residual_vs_phi_vs_eta(electrons_df,axis,super_module,charge,range_y):    
    electrons_sm = electrons_df[(electrons_df['super_module_number'] == super_module) & (electrons_df['charge'] == charge) ]
    fig, ax = plt.subplots()
    if (range_y == None):
            counts, xedges, yedges, im  = ax.hist2d(electrons_sm['phi_cluster'], electrons_sm[str(axis)],norm=LogNorm(),bins=(10,50))
    else:
        counts, xedges, yedges, im  = ax.hist2d(electrons_sm['phi_cluster'], electrons_sm[str(axis)],norm=LogNorm(),bins=(10,50),range=[(electrons_sm['phi_cluster'].min(),electrons_sm['phi_cluster'].max()),range_y])
    ax.set_ylabel("Residual in "+str(axis))
    ax.set_xlabel(r"$\varphi$")
    fig.colorbar(im,ax=ax)
    fig.savefig("res_vs_phi_sm_"+str(super_module)+ "_axis_"+axis+"_sign_"+str(charge)+".svg",bbox_inches='tight')
    return ax


def plot_residual_vs_zvtx(electrons_df,axis,super_module,charge,range_y=None):    
    electrons_sm = electrons_df[(electrons_df['super_module_number'] == super_module) & (electrons_df['charge'] == charge) ]
    fig, ax = plt.subplots()
    if (range_y == None):
            counts, xedges, yedges, im  = ax.hist2d(electrons_sm['zvtx'], electrons_sm[str(axis)],norm=LogNorm(),bins=(10,50))
    else:
        counts, xedges, yedges, im  = ax.hist2d(electrons_sm['zvtx'], electrons_sm[str(axis)],norm=LogNorm(),bins=(10,50),range=[(electrons_sm['zvtx'].min(),electrons_sm['zvtx'].max()),range_y])
    ax.set_ylabel("Residual in "+str(axis))
    ax.set_xlabel(r"$Z_{vtx}$")
    fig.colorbar(im,ax=ax)
    fig.savefig("res_vs_zvtx_sm_"+str(super_module)+ "_axis_"+axis+"_sign_"+str(charge)+".svg",bbox_inches='tight')
    return ax

plot_residual_vs_zvtx(electrons,'z_res',0,1,range_y=(-10,10))


plot_residual_vs_phi_vs_eta(electrons,'y_res',0,1,(-10,10))


plot_residual_vs_phi_vs_eta(electrons,'y_res',16,1,(-10,10))


def plot_basic_qa():
    fig_pid, ax_pid = plt.subplots()
    ax_pid = sns.distplot(electrons['e_over_p'],axlabel = 'E/p',kde=False)
    ax_pid.set_ylabel("Counts")
    fig_pid.savefig("e_over_p.png",bbox_inches='tight')
    
    fig_pt, ax_pt = plt.subplots()
    ax_pt = sns.distplot(electrons['pt'],axlabel=r'$p_T$ GeV/c',kde=False)
    ax_pt.set_ylabel("Counts")
    fig_pt.savefig("pt.svg",bbox_inches='tight')
    
    fig_eta, ax_eta = plt.subplots()
    ax_eta = sns.distplot(electrons['eta_cluster'],axlabel=r'$\eta_{cluster}$',kde=False)
    ax_eta.set_ylabel("Counts")
    fig_eta.savefig("eta.svg",bbox_inches='tight')
    
    fig_phi, ax_phi = plt.subplots()
    ax_phi = sns.distplot(electrons['phi_track'],axlabel=r'$\varphi^{track}$',kde=False)
    ax_phi.set_ylabel("Counts")
    fig_phi.savefig("phi_track.svg",bbox_inches='tight')
    
    fig, ax = plt.subplots()
    pal = sns.cubehelix_palette(8,as_cmap=True)
    counts, xedges, yedges, im  = ax.hist2d(electrons['phi_cluster'], electrons['eta_cluster'],bins=(100,100),cmap=pal)
    ax.set_ylabel(r"$\eta_{cluster}$")
    ax.set_xlabel(r"$\varphi_{cluster}$")
    fig.colorbar(im,ax=ax)
    fig.savefig("cluster_map.svg",bbox_inches='tight')
    
    fig_phi, ax_phi = plt.subplots()
    counts, xedges,  im  = ax_phi.hist(electrons['phi_cluster'],bins=2000)

    ax_phi.set_ylabel("Counts")
    ax_phi.set_xlabel(r'$\varphi^{cluster}$')
    fig_phi.savefig("phi_cluster_fine_bin.svg",bbox_inches='tight')
    
    fig_phi, ax_phi = plt.subplots()
    counts, xedges,  im  = ax_phi.hist(electrons['eta_cluster'],bins=200)

    ax_phi.set_ylabel("Counts")
    ax_phi.set_xlabel(r'$\varphi^{cluster}$')
    fig_phi.savefig("eta_cluster_fine_bin.svg",bbox_inches='tight')
    
    fig_phi, ax_phi = plt.subplots()
    counts, xedges,  im  = ax_phi.hist(electrons['phi_track'],bins=2000)

    ax_phi.set_ylabel("Counts")
    ax_phi.set_xlabel(r'$\varphi^{track}$')
    fig_phi.savefig("phi_tracks_fine_bin.svg",bbox_inches='tight')
    
    fig_phi, ax_phi = plt.subplots()
    ax_phi = sns.distplot(electrons['n_sigma_electron_TPC'],axlabel=r'n$\sigma_{TPC}^{e}$',kde=False)
    ax_phi.set_ylabel("Counts")
    fig_phi.savefig("n_sigma_electron_TPC.png",bbox_inches='tight')


for sm in range(0,20):
    
    fig_phi, ax_phi = plt.subplots()
    counts, xedges,  im  = ax_phi.hist(electrons['phi_cluster'][electrons['super_module_number'] == sm],bins=10)

    ax_phi.set_ylabel("Counts")
    ax_phi.set_xlabel(r'$\varphi^{cluster}$')
    ax_phi.set_title("SM "+ str(sm))
    fig_phi.savefig("phi_cluster"+str(sm)+".svg",bbox_inches='tight')


values = electrons.groupby(['super_module_number', 'charge']).mean()
errors = electrons.groupby(['super_module_number', 'charge']).std()/np.sqrt(electrons.groupby(['super_module_number', 'charge']).count())
errors_percent = np.absolute(errors/values)


sns.set_palette("hls", 2)

plot_y =values['phi_res'].unstack()
plot_e = errors['phi_res'].unstack()
fig, ax = plt.subplots()
ax.errorbar(range(0,18), y=plot_y[-1], yerr=plot_e[-1],fmt='o',label='-1')
ax.errorbar(range(0,18), y=plot_y[1], yerr=plot_e[1],fmt='o',label='1')
ax.set_xlim(-1,20)
ax.legend(title='Charge')
ax.set_xlabel("Super module number")
ax.set_ylabel(r"Mean $\Delta \varphi^{electron}$ ")
fig.savefig("phi_mean_per_sm_per_charge_with_error.png",bbox_inches='tight',dpi=1200)


plot_y =values['eta_res'].unstack()
plot_e = errors['eta_res'].unstack()
fig, ax = plt.subplots()
ax.errorbar(range(0,18), y=plot_y[-1], yerr=plot_e[-1],fmt='o',label='-1')

ax.errorbar(range(0,18), y=plot_y[1], yerr=plot_e[1],fmt='o',label='1')

ax.set_xlim(-1,20)
ax.legend(title='Charge')
ax.set_xlabel("Super module number")
ax.set_ylabel(r"Mean $\Delta \eta^{electron}$ ")
fig.savefig("eta_mean_per_sm_per_charge_with_error.png",bbox_inches='tight',dpi=1200)


def plot_mean_res(variable):
    plot_y =values[variable].unstack()
    plot_e = errors[variable].unstack()
    fig, ax = plt.subplots()
    
    ax.errorbar(range(0,18), y=plot_y[-1], yerr=plot_e[-1],fmt='o',label='-1')
    ax.errorbar(range(0,18), y=plot_y[1], yerr=plot_e[1],fmt='o',label='1')

    ax.set_xlim(-1,20)
    ax.legend(title='Charge')
    ax.set_xlabel("Super module number")
    ax.set_ylabel(r"Mean $\Delta "+str(variable[0])+"^{electron}$ ")
    fig.savefig(str(variable)+"_mean_per_sm_per_charge_with_error.png",bbox_inches='tight',dpi=1200)
    
plot_mean_res('x_res')
plot_mean_res('y_res')
plot_mean_res('z_res')

def plot_all_sm_in_one_plot(range_sm,charge,variable,name_to_save):
    fig,ax = plt.subplots(figsize=(0.9*8,0.9*6))
    for sm in range_sm:
        electrons_sm = electrons[(electrons['super_module_number'] == sm)&(electrons['charge'] == charge)]
        electrons_sm[variable+'_res'].hist(bins=100,ax=ax,label=str(sm),range=(-0.1,0.1),histtype='step',linewidth=2.5,grid=False)
    ax.set_yscale("log")
    ax.set_ylabel("Counts")
    if variable == 'phi':
        ax.set_xlabel(r"$\Delta\varphi$ residual")
    else:
        ax.set_xlabel(r"$\Delta\eta$ residual")
    name_to_save = variable+'_'+name_to_save
    
    if charge == 1:
        ax.legend(title="Positive",ncol=2)
    else:
        ax.legend(title="Negative",ncol=2)
    fig.savefig(name_to_save,bbox_inches='tight')


sns.set_palette("hls", 12)


plot_all_sm_in_one_plot(range(0,12),1,'phi',"emcal_positive.png")


plot_all_sm_in_one_plot(range(0,12),1,'eta',"emcal_positive.png")


plot_all_sm_in_one_plot(range(0,12),-1,'phi', "emcal_negative.png")


plot_all_sm_in_one_plot(range(0,12),-1,'eta', "emcal_negative.png")


sns.set_palette("hls", 6)
plot_all_sm_in_one_plot(range(12,18),-1,'phi', "dcal_negative.png")

plot_all_sm_in_one_plot(range(12,18),-1,'eta', "dcal_negative.png")


plot_all_sm_in_one_plot(range(12,18),1,'phi', "dcal_positive.png")


plot_all_sm_in_one_plot(range(12,18),1,'eta', "dcal_positive.png")