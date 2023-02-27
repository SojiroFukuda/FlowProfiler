
import flowprofiler as fp
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

text_format = ".txt"

####
##  exp class: container of fp.run_profiler() objects.
####
class exp(object):
    # Constructor
    def __init__(self, name):
        super(exp, self).__init__()
        self.name = name        # paper
        self.run_dict = {}      # container of run objects
    # assign new run object
    def add_run(self,run):
        # exec('{}={}'.format("self.run_"+str(run.number),"run")) 
        self.run_dict[str(run.number)] = run                # add run object into dict
        self.run_dict[str(run.number)].paper = self.name    # assign paper name for run object
    # return run object from dict
    def get_run(self,number):
        return self.run_dict.get(str(number))

### Import exp data from Folda
##  exp_name must correspond the experiment name
##  exp_info must contain all setting data of all paper
##  coord_folda must contain .txt files of flow profiles
####

def initiate_runobj_from_meta_info(run_num,info):
    """ return fp.run_profiler obejcts, assigning the info given by the pd.DataFrame object.

    Args:
        run_num (str): name of run
        info (pd.DataFrame): DataFrame object where the info of the run is kept.

    Returns:
        run_obj (fp.run_profiler): the run object with basic info of its setting.
        densitytype (str): string name of the format of concentration unit (see density_keys)
        
    """
    density_keys = {
        'SW':'sedimentWeight',
        'FD':'flowDensity',
        'VC':'volumetricConcentration',
        'FED':'fractionalExcessDensity',
        'BY':'Buoyancy'
        } #Density profile type#
    SW = 0 # sediment weight: sediment weight per a specific volume in flow [kg/m^3]
    FD = 1 # Flow Density [kg/m3]
    VC = 2 # Volumetric concentration of sediment
    FED = 3 # Fractional excess density
    BY = 4 # Buoyancy B=RC

    # extract info for the given run
    if run_num.isdigit():
        run_info = info[info['run']==str(int(run_num))] # run number to string
    else:
        run_info = info[info['run']==run_num] 
    # exclude the data with zero slope
    if float( np.squeeze(run_info['slope(tan)']) ) == 0:
        return np.nan, np.nan
    else:
        # initiate run object ------------
        density_type = density_keys[str(info['densitytype'].iloc[0])]
        run_obj = fp.run_profiler(run_num) # Number of Run
        run_obj.Cprofile_available = bool(run_info.Cprofile_available.iloc[0]) # whether density profile is available or not
        run_obj.isConfinedChannel = bool(run_info.Confined.iloc[0] == 'Fully' )
        run_obj.bh = float(run_info['Bed_height']) # bed height
        run_obj.slope = float(run_info['slope(tan)']) # Slope
        if run_info.flume_wide.iloc[0] != None:
            run_obj.flumeWidth = float(run_info['flume_wide'])/100 # width of flume (meter) 
        # run_obj.R = float(run_info.R)
        if run_info.Init_Conc.iloc[0] != None:
            run_obj.Czero = float(run_info['Init_Conc']) # Concentration in Mixing tank
        run_obj.rhos = float(run_info['Rhos(kgm)']) # density of the sediment partcle [kg/m^3]
        run_obj.rhoa = float(run_info['Rhoa(kgm)']) # density of ambient water [kg/m^3]
        run_obj.d = float(run_info['d50 (micron)']) * np.power(10.0,-6) # 50 percentail grain size d50 [m] 
        run_obj.m_type = str(run_info.mu.iloc[0])  
        run_obj.m = fp.mu(mu_type=run_obj.m_type, c_ave=-1, rhos=-1)
        run_obj.isConservative = str(run_info.Conservative.iloc[0])
        run_obj.particle_type = str(run_info.Particle_type.iloc[0])
        run_obj.submarine = str(run_info.submarine.iloc[0])
        return run_obj, density_type

def set_velocity_profile(run_obj, velocity_path,iscm=True):
    """velocity profile interpolation/extrapolation handler.

    Args:
        run_obj (fp.run_profiler): run object
        velocity_path (str): file path of the text data where the original velocity measurement data are kept.
        iscm (bool, optional): the unit of length of the original data (meter or centi meter). Set False if the unit is meter. Defaults to True.
    """
    data_v = pd.read_table(velocity_path) # import original measurement
    vx_pchip, vy_pchip, vx, vy, vx_up, vy_up, vym_ori = fp.velocity_profiler(data_v,run_obj.bh,islogmode=False,iscm=iscm) # interpolation / extrapolation
    run_obj.vx = vx_pchip # interpolated x
    run_obj.vy = vy_pchip # interpolated y
    run_obj._vx = vx # original x
    run_obj._vy = vy # original y
    run_obj.hv = max(run_obj.vy) # height estimated from velocity profile
        
def set_concentration_profile(run_obj,PATH_C,density_type,iscm=True):
    """concentration prpfile interpolation/extrapolation handler

    Args:
        run_obj (fp.run_profiler): run object
        PATH_C (str): file path of the text data where the original concentration measurement data are kept.
        density_type (str): type of density data (volumetric concentration, fractional excess density, etc.)
        iscm (bool, optional):the unit of length of the original data (meter or centi meter). Set False if the unit is meter.  Defaults to True.
    """
    data_c = pd.read_table(PATH_C) # read original measurement
    if density_type == 'flowDensity': # if the density type is flow density [kg/m3]
        min_value = run_obj.rhoa # set background density
        
        # subtract the background density from the original measurement data
        data_cf = pd.DataFrame() 
        data_cf['x'] = data_c['x'] - min_value
        data_cf['y'] = data_c['y']
        
        # interpolation/extrapolation
        cx_all, cy_all, cx_pr, cy_pr, cy_up, cx, cy, cym_ori = fp.concentration_profiler(data_cf,run_obj.bh,0,run_obj.isConservative,islogmode=False,iscm=iscm)
        cx_all = cx_all + min_value # interpolated x
        cx_pr = cx_pr + min_value # 
        cx = cx + min_value # original
    else:
        # interpoaltion/extrapolation
        min_value = 0
        cx_all, cy_all, cx_pr, cy_pr, cy_up, cx, cy, cym_ori = fp.concentration_profiler(data_c,run_obj.bh,min_value,run_obj.isConservative,islogmode=False,iscm=iscm)
    # assign profiles to the run object
    run_obj.cx = cx_all
    run_obj.cy = cy_all
    run_obj._cx = cx
    run_obj._cy = cy
    run_obj.hc = cy_up
    run_obj.min_value = min_value
    

def set_flowheight(run):
    """flow height handler. Estimate the flow height based upon the heights estimated from velocity and concentration profile respectively.
    
    Args:
        run (fp.run_profiler):run object (after interpolation/extrapolation procedures)
    """
    # temp containers
    vx_mod = []; vy_mod = []
    cx_mod = []; cy_mod = []
    
    if run.hv <= run.hc: # when velocity height is lower than the concentration height
        f = interpolate.interp1d(run.cy, run.cx, kind='linear')
        f = interpolate.PchipInterpolator(run.cy,run.cx)
        cy_mod = np.linspace(0,run.hv,num=500)
        cx_mod = f(cy_mod)
        vx_mod = run.vx
        vy_mod = run.vy
    elif run.hv > run.hc: # when velocity height is higher than the concentration height
        if run.hc <= np.max(run._vy): # when concentration height is lower than the highest non-zero original velocity point.
            cx_low = run.cx[0]
            cy_low = run.cy[0]
            cx_high = run.min_value
            cy_high = run.hv
            if run._cy[0] != run.cy[0]: # add near-bed conncentration
                cx_mod = np.append(cx_low,run._cx)
                cy_mod = np.append(cy_low,run._cy)
            else:
                cx_mod = run._cx
                cy_mod = run._cy
            cx_mod = np.append(cx_mod,cx_high)
            cy_mod = np.append(cy_mod,cy_high)
            cx_mod,cy_mod = fp.profile_interpolate(cx_mod,cy_mod,500)
            vx_mod = run.vx
            vy_mod = run.vy
        else: # when concentration height is higher than the highest non-zero original velocity point but lower than the velocity height
            for i,val in enumerate(run._vy):
                if val < run.hc:
                    vx_mod.append(run._vx[i])
                    vy_mod.append(run._vy[i])
            vx_mod.append(0)
            vy_mod.append(run.hc)
            cx_mod = run.cx
            cy_mod = run.cy
            vx_mod,vy_mod = fp.profile_interpolate(vx_mod,vy_mod,500)
    run.vx = vx_mod; run.vy = vy_mod
    run.cx = cx_mod; run.cy = cy_mod
    

def set_depth_averaged_params(run,density_type):
    # velocity
    run.U_ave = fp.U_ave(run.vx,run.vy,run.height) 
    # concentration
    if run.Cprofile_available:
        if density_type == 'sedimentWeight':
            # print("Original Data: sediment concentration [kg/m^3]")
            run.sedimentWeight(run.cx, run.cy, run._cx, run._cy)
        elif density_type == 'flowDensity':
            # print("Original Data: flow density (rho_f) [kg/m^3]")
            run.flowDensity(run.cx, run.cy, run._cx, run._cy)
        elif density_type == 'volumetricConcentration':
            # print("Original Data: volumetric concentration [-]")
            run.volumetricConcentration(run.cx, run.cy, run._cx, run._cy)
        elif density_type == 'fractionalExcessDensity':
            # print("Fractional excess density (rho_f - rho_a) [kg/m^3]")
            run.fractionalExcessDensity(run.cx, run.cy, run._cx, run._cy)
        elif density_type == 'Buoyancy':
            run.buoyancy(run.cx, run.cy, run._cx, run._cy)
        else:
            raise Exception("ERROR: The unit of concentration is wrongly assigned. Check density_type in the code.") 
    else: # For the data where only initial concentration data is available (TYPE II)
        if density_type == 'sedimentWeight':
            run.set_C_ave(run.Czero)
        elif density_type == 'flowDensity':
            run.set_Rhof_ave(run.Czero)
        elif density_type == 'volumetricConcentration':
            run.set_Cv_ave(run.Czero)
        elif density_type == 'fractionalExcessDensity':
            run.set_Rhof_ave((run.Czero*run.rhoa)+run.rhoa)
        elif density_type == 'Buoyancy':
            run.set_Rhof_ave((run.Czero*run.rhoa)+run.rhoa)
        else:
            raise Exception("ERROR: The unit of concentration is wrongly assigned. Check density_type in the code.") 
    # buoyancy
    run.B = ( run.rhof_ave - run.rhoa ) / run.rhoa

    
def set_exp(exp_name: str, exp_info: str, coord_folda: str, FORMAT_PROFILES: str = '.txt', iscm: bool = True) -> exp:
    """
    Data compilation based on the profile data and other meta data.
    The Data compilation is conducted within this function except for the calculation of shear velocity and Congo's event data.
    For the calculation of shear velocity, see ShearVelocity.ipynb. For the Congo date, see Congo_profiler.ipynb.
    It returns main.exp() object which stores main.run() objects. Each run object corresponds to each run of the experiment.

    Parameter
    -----
    exp_name: str
        The key of each source. See keys of PaperDict in database.py for the detaield key for each source.
    exp_info: str
        The path for the csv file which stores the meta data of each source. See database.py for the file path.
    coord_folda: str
        The path of the directory where the profiles data are kept as text files. See database.py for the path.
    FORMAT_PROFILES: str = '.txt'
        The file extension of profile data in coord_folda. By default, it is set as .txt.
    iscm: bool = True
        The unit of the original velocity profile. Set it as True if the unit is centi meter. If it is meter, then set it as False.
    """
    # key for the unit of concentration data
    # initiate data object
    exps = exp(exp_name) # initiate experiment object
    # load meta data from csv file
    info = pd.read_csv(exp_info).loc[pd.read_csv(exp_info)['Exp']==exp_name] # load CSV of meta data
    # iterates procedures through velocity profile coordinates
    for file in fp.velocity_list(coord_folda,format=FORMAT_PROFILES): # sort .txt file where velocity profile info is kept
        file_array = file.split('_') # .txt file has a name in the format: run_[number]_velocity.txt / run_[number]_concentration.txt
        run_num = str(file_array[1]) # extract run name from the file name
        run_obj, density_type = initiate_runobj_from_meta_info(run_num,info) # import meta data from csv file
        
        # velocity data
        PATH_V = str( coord_folda + os.sep + 'run_' + run_num + '_' + 'velocity' + FORMAT_PROFILES ) # path of velocity profiles
        set_velocity_profile(run_obj,PATH_V,iscm=iscm) # interpolation & extrapolation
        
        # Concentration data
        if run_obj.Cprofile_available == True: 
            PATH_C = str( coord_folda + os.sep + 'run_' + run_num + '_' + 'concentration' + FORMAT_PROFILES ) # path of concentration profiles
            set_concentration_profile(run_obj,PATH_C,density_type,iscm=iscm) # interpoaltion & extrapolation
            set_flowheight(run_obj) # set flow height and re-interpret the flow profiles based on the new flow height
        run_obj.height = np.max(run_obj.vy) # assign flow height to run object (set_flowheight() method adjust the parameters so that run_obj.height == np.max(run_obj.vy) for the data where concentration profile is available)
        set_depth_averaged_params(run_obj,density_type) # Volumetric concentration, flow density, fractional excess denstiy, etc.
        exps.add_run(run_obj) # store run object
    return exps


def draw_profile(exps,save_dir):
    """
    Draw normalised streamwise flow velocity and flow concentration profiles from the given set of exp objects.
    
    """
    # print(exps.name)
    fig_dir = save_dir+os.sep+"Profiles"+os.sep+exps.name
    os.makedirs(fig_dir,exist_ok=True) # Save Folda
    plt.close()
    import figformat as ff
    A4 = ff.get_A4size()
    for run in exps.run_dict.values():
        fig = plt.figure(figsize=(A4['width'],A4['half']))
        axes = fig.subplots(1,2)
        axes[0].plot(run.vx/run.U_ave, run.vy/run.height,color='gray')
        axes[0].scatter(run._vx/run.U_ave, run._vy/run.height)
        axes[0].grid(visible=True)
        if run.Cprofile_available:
            axes[1].plot(run.rhofx/run.rhof_ave , run.rhofy/run.height,color='gray')
            axes[1].scatter(run._rhofx/run.rhof_ave, run._rhofy/run.height)
            axes[1].grid(visible=True)
        else:
            axes[1].text(0.2,0.5,'Data unavailable',transform=axes[1].transAxes,fontsize=A4['FSIZE'])
        axes[0].set_title('Normalised streamwise velocity', fontsize=A4['FSIZE'])
        axes[1].set_title('Normalised flow density',  fontsize=A4['FSIZE'])
        axes[0].set_xlabel(r'$\langle u \rangle / U$', fontsize=A4['FSIZE'])
        axes[0].set_ylabel(r'$z / h$', fontsize=A4['FSIZE'])
        axes[1].set_xlabel(r'$ h \langle \rho \rangle / \int \langle \rho \rangle \mathrm{d}z$', fontsize=A4['FSIZE'])
        plt.savefig(fig_dir+os.sep+"run_"+str(run.number)+".pdf",bbox_inches='tight')
        plt.close()
    # print(exps.name+str(":end"))



# export summary of experiment
def export_results(exps,save_dir):
    """Export the compiled results to the given path.
    
    Args:
        exps (main.exp): container of fp.run_profiler obejcts
        save_dir (str): save path of the results.
    returns:
        parameter_df (pd.DataFrame): DataFrame object of the depth-averaged flow params.
    """
    os.makedirs(save_dir+os.sep+"Summary",exist_ok=True) # Save Folda
    os.makedirs(save_dir+os.sep+"Profile_txt",exist_ok=True) # Save Folda
    os.makedirs(save_dir+os.sep+"Parameter_txt",exist_ok=True) # Save Folda
    PROFILE_PATH = save_dir+os.sep+"Profile_txt"
    PARAMETER_PATH = save_dir+os.sep+"Parameter_txt"

    for r in exps.run_dict.values():
        #==== Export Profile data =======#
        profile_df = pd.DataFrame()
        profile_df['velocity'] = r.vx/r.U_ave # normalised interpolated and extrapolated velocity
        vx_array = np.ones(500) * (-1.0)
        hv = np.array((r._vy[ (r._vy/r.height)*499 < 500 ] / r.height) * 499, dtype=np.int16)
        vx_array[hv] = r._vx[ (r._vy/r.height)*499 < 500 ] / r.U_ave
        profile_df['vel_ori'] = vx_array # original measurement points
        if r.Cprofile_available:
            profile_df['density'] = (r.rhofx-r.rhoa)/(r.rhof_ave-r.rhoa)
            cx_array = np.ones(500) * (-1.0)
            hc = np.array((r._rhofy[(r._rhofy/r.height)*499 < 500 ] / r.height) * 499, dtype=np.int16)
            cx_array[hc] = (r._rhofx[(r._rhofy/r.height)*499 < 500]-r.rhoa)/(r.rhof_ave-r.rhoa)
            profile_df['den_ori'] =  cx_array
        else:
            profile_df['density'] = np.ones(500) * -1
            profile_df['den_ori'] = np.ones(500) * -1
        profile_df['eta'] = np.linspace(0,1,500)
        profile_df.to_csv(PROFILE_PATH+os.sep+str(exps.name)+'_'+str(r.number)+'.txt',header=True)
        #==== -------------------- =======#
        
        parameter_df = pd.DataFrame()
        parameter_df['isConservative'] = np.array([r.isConservative])
        parameter_df['Czero'] = np.array([r.Czero])
        parameter_df['U_ave'] = np.array([r.U_ave])
        parameter_df['rhof_ave'] = np.array([r.rhof_ave])
        parameter_df['rhoa'] = np.array([r.rhoa])
        parameter_df['height'] = np.array([r.height])
        parameter_df['HR'] = np.array([((r.height*r.flumeWidth)/(2*r.height + r.flumeWidth))/r.height])
        parameter_df['Frd'] = np.array([r.U_ave/np.sqrt(r.B*r.height*r.g)])
        parameter_df['Re'] = np.array([r.U_ave*r.height*r.rhof_ave/0.001])
        parameter_df['Ri'] = np.array([r.height*r.g*r.B/np.power(r.U_ave,2) ])
        parameter_df['B'] = np.array([r.B])
        parameter_df['slope'] = np.array([r.slope])
        parameter_df['m'] = np.array([r.m])

        if r.isConservative == 'Non-Conservative':
            parameter_df['rhos'] = np.array([r.rhos])
            parameter_df['ws'] = np.array([r.ws])
        else:
            parameter_df['rhos'] = np.array([0])
            parameter_df['ws'] = np.array([0])
        parameter_df['Cprofile'] = np.array([r.Cprofile_available])
        
        if r.isConservative == 'Non-Conservative':
            parameter_df['d50'] = np.array([r.d])
        elif r.isConservative == 'Conservative':
            parameter_df['d50'] = np.array([0])
        parameter_df.to_csv(PARAMETER_PATH+os.sep+str(exps.name)+'_'+str(r.number)+'.txt',header=True)
    return parameter_df




