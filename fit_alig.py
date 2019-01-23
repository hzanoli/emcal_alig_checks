import ROOT
import root_numpy
import numpy as np
import pandas as pd
ROOT.TH1.AddDirectory(False)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.colors import LogNorm
import scipy as sp
sns.set_context("talk")


#import data from ROOT file
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


#cuts - select the electrons with resonable cuts
min_TPC_nsig = -1.0
max_TPC_nsig = 3.0
eta_res_cut = 0.1
phi_res_cut = 0.1
e_over_p_cut_min = 0.9
e_over_p_cut_max = 1.2
res_max = 900
pt_min = 3.0
pt_max = 10

eta_clus_min = 0
eta_clus_max = 0.7

#pT cut
electrons['p'] = np.sqrt(electrons['pt']**2+electrons['pz']**2)
electrons['e_over_p'] = electrons['energy']/electrons['p']

is_pt = (electrons['pt'] >= pt_min) & (electrons['pt'] <= pt_max)
electrons = electrons[is_pt]

#E/p cut
is_e_over_p = (electrons['e_over_p'] >= e_over_p_cut_min) & (electrons['e_over_p'] <= e_over_p_cut_max)

#eta_clus_cut
is_eta_cut = (np.absolute(electrons['eta_cluster'])> eta_clus_min) & (np.absolute(electrons['eta_cluster']) < eta_clus_max)

#other cuts
is_TPC_PID = (electrons['n_sigma_electron_TPC'] >= min_TPC_nsig) & (electrons['n_sigma_electron_TPC'] <= max_TPC_nsig)
electrons = electrons[is_TPC_PID & is_e_over_p & is_eta_cut]

#Eta (the pseudorapitity) is not implemented on numpy. Short implementation
def Eta(x,y,z):
    cos_theta = z/np.sqrt(x**2 + y ** 2 + z ** 2)
    eta = -0.5* np.log((1.0-cos_theta)/(1.0+cos_theta) )
    return eta

#Calculate the position of the tracks on the calorimeter
electrons['phi_track_emc'] = np.arctan2(electrons['y_track'],electrons['x_track'])
electrons['eta_track_emc'] = Eta(electrons['x_track'], electrons['y_track'], electrons['z_track'])

#Calculate the trend of the matching to the calorimeter
def calculate_residuals(df):
    df['x_res'] = df['x_track'] - df['x_cluster']
    df['y_res'] = df['y_track'] - df['y_cluster']
    df['z_res'] = df['z_track'] - df['z_cluster']
    df['phi_res'] = df['phi_track_emc'] - df['phi_cluster']
    df['eta_res'] = df['eta_track_emc'] - df['eta_cluster']
    
calculate_residuals(electrons)

#remove very bad matches
electrons = electrons[np.abs(electrons['phi_res'])<0.05]
electrons = electrons[np.abs(electrons['eta_res'])<0.05]

#implement the rotation matrix as in ALICE
def rotation_matrix(psi, theta, phi):
    'angles in degrees, copied from the ALICE convetion'
    psi = np.radians(psi)
    theta = np.radians(theta)
    phi = np.radians(phi)
    rot = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.])

    sinpsi = np.sin(psi)
    cospsi = np.cos(psi)
    sinthe = np.sin(theta)
    costhe = np.cos(theta)
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    
    rot[0] =  costhe*cosphi
    rot[1] = -costhe*sinphi
    rot[2] =  sinthe
    rot[3] =  sinpsi*sinthe*cosphi + cospsi*sinphi
    rot[4] = -sinpsi*sinthe*sinphi + cospsi*cosphi
    rot[5] = -costhe*sinpsi
    rot[6] = -cospsi*sinthe*cosphi + sinpsi*sinphi
    rot[7] =  cospsi*sinthe*sinphi + sinpsi*cosphi
    rot[8] =  costhe*cospsi
    
    return rot.reshape(3,3)

#apply the translation and the rotation at the same time
def translation_rotation(X,cluster_postions,track_postions):
    rot = rotation_matrix(X[0],X[1],X[2])
    translation = np.array([X[3],X[4],X[5]])
    residuals = np.array((np.dot(cluster_postions,rot)+translation)-track_postions)
    residuals = np.sqrt((residuals**2).sum(axis=1))
    return residuals

results_of_opt = list()

for sm in range(0,18):
    electrons_sm = electrons[electrons['super_module_number']==sm].copy()
    electrons_sm = electrons_sm[electrons_sm['charge']>0].copy()
    track_postions = np.array(pd.DataFrame([electrons_sm['x_track'],electrons_sm['y_track'],electrons_sm['z_track']]).T)
    cluster_postions = np.array(pd.DataFrame([electrons_sm['x_cluster'],electrons_sm['y_cluster'],electrons_sm['z_cluster']]).T)


    initial_guess = (1.0,1.0,1.0,electrons_sm['x_res'].mean(),electrons_sm['y_res'].mean(),electrons_sm['z_res'].mean())
    OptimizeResult = sp.optimize.least_squares(translation_rotation,initial_guess,args=(cluster_postions,track_postions))
    results_of_opt.append(OptimizeResult)
    
    print(OptimizeResult.x)

    no_alig = [0.,0.,0.,0.,0.,0.]

    fig, ax = plt.subplots()
    points = translation_rotation(no_alig,cluster_postions,track_postions)
    ax.hist(points,bins=100)
    ax.set_yscale('log')

    fig, ax = plt.subplots()
    points = translation_rotation(OptimizeResult.x,cluster_postions,track_postions)
    ax.hist(points,bins=100)
    print((points**2).sum())
    ax.set_yscale('log')


    def alig_clusters(X,cluster_postions):
        rot = rotation_matrix(X[0],X[1],X[2])
        translation = np.array([X[3],X[4],X[5]])
        new_positions = np.dot(cluster_postions,rot)+translation
        return new_positions

    calculate_residuals(electrons_sm)

    temp_p = alig_clusters(OptimizeResult.x,cluster_postions)
    frame = pd.DataFrame(temp_p)
    electrons_sm = electrons[electrons['super_module_number']==sm].copy()
    electrons_sm['x_track'] = frame[0]
    electrons_sm['y_track'] = frame[1]
    electrons_sm['z_track'] = frame[2]


    values = electrons_sm.groupby(['super_module_number', 'charge']).mean()
    errors = electrons_sm.groupby(['super_module_number', 'charge']).std()/np.sqrt(electrons_sm.groupby(['super_module_number', 'charge']).count())
    errors_percent = np.absolute(errors/values)
