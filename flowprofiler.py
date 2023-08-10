import numpy as np
import os
import re
from scipy.optimize import curve_fit
from scipy.integrate import trapz
from scipy import interpolate
import csv
import pandas as pd


class run_profiler(object):
    """Run object which store the vertical profile information and other meta data of each run."""
    def __init__(self, number: str = '') -> None:
        """Run object
        run_profiler() provides an object which store and calculate the vertical flow profiles and related flow parameters.

        Args:
            number (str): name of each run.
        """
        super(run_profiler, self).__init__()
        self.number = number        # run number
        # basic constants
        self.g = 9.8 # gravity acceleration 
        self.Karman = 0.4 # von Karman constant
        
    def set_id(self, file_name: str, PaperDict: dict) -> None:
        """
        Parameter
        -----
        file_name: str
            The file name of the .txt file created by the main.export_results(). It has the format: [experiment name]_[run number].txt.
        PaperDict: dict
            The dictionary of the reference of each source. (see database.py)
        """
        ids = file_name.split('_')
        self.exp = ids[0] # experiment name
        self.paper_name = PaperDict[self.exp] # reference
        self.number = ids[1] # number of each run
        self.id = self.exp+'_'+self.number # identical key for each run object

    def load_profiles(self,path: str) -> None:
        """
        Parameter
        -----
        path: str 
            The path of the .txt file where the vertical flow profiles are kept. The .txt file is created by main.export_results().
            See main.py for the detail format of this .txt file. 
        """
        if path == '':
            pass
        else:
            eta = 'eta'
            vx_ori = 'vel_ori'
            cx_ori = 'den_ori'
            vx_calc = 'velocity'
            cx_calc = 'density'
            # store original data
            self.data = pd.read_csv(path)
            # Measured Points #
            self.velocity_profile = pd.DataFrame()
            self.concentration_profile = pd.DataFrame()
            self.velocity_measurements = pd.DataFrame()
            self.concentration_measurements = pd.DataFrame()
            # quick access to the original measured values 
            self._vx = self.data[vx_ori][ self.data[vx_ori] > -1 ]
            self._vy = self.data[eta][ self.data[vx_ori] > -1 ]
            self._cx = self.data[cx_ori][ self.data[cx_ori] > -1 ] # Buoyancy
            self._cy = self.data[eta][ self.data[cx_ori] > -1 ] # Buoyanvy
            # assign DataFrame object for velocity and concentration
            self.velocity_measurements["x"] = self._vx; self.velocity_measurements["y"] = self._vy
            self.concentration_measurements["x"] = self._cx; self.concentration_measurements["y"] = self._cy
            # Inter/Extrapolated profile #
            self.vx = self.data[vx_calc]
            self.vy = self.data[eta]
            self.cx = self.data[cx_calc]
            self.cy = self.data[eta]
            self.velocity_profile["x"] = self.vx; self.velocity_profile["y"] = self.vy
            self.concentration_profile["x"] = self.cx; self.concentration_profile["y"] = self.cy
        
        
    def load_parameter(self, path: str) -> None:
        """
        Import all meta data from txt file which is exported by main.export_results().
        
        Parameter
        -------
        path: str
            the filepath of the meta data exported by main.expoert_results(). They should be kept in data/TC/Parameter_txt/.
        """
        import pandas as pd
        info = pd.read_csv(path) # read .txt file
        # parameters
        self.isConservative = str(info['isConservative'][0]) == "Conservative" # whether the run is conservative (bool)
        self.Czero = info['Czero'].values.astype(float)[0]  # init concentration in mixing tank
        self.U_ave = info['U_ave'].values.astype(float)[0]  # depth-averaged flow velocity [m/s]
        self.rhof_ave = info['rhof_ave'].values.astype(float)[0]    # depth-averaged flow density [kg/m^3]
        self.rhoa = info['rhoa'].values.astype(float)[0]    # density of ambient fluid [kg/m3]
        self.rhos = info['rhos'].values.astype(float)[0]    # density of sediment particle [kg/m3]
        self.R = (self.rhos - self.rhoa)/self.rhoa  # Specific gravity
        self.height = info['height'].values.astype(float)[0]    # flow height [m]
        self.B = info['B'].values.astype(float)[0]  # depth-averaged flow Buoyancy
        self.Cv_ave = self.B/self.R # depth-averaged volumetric flow concentration
        self.slope = info['slope'].values.astype(float)[0]  # Slope (tangent)
        self.m = info['m'].values.astype(float)[0]  # dynamic viscosity of flow [kg/ms]
        self.nu = self.m / self.rhof_ave # kinematic viscosity [kg*m2/s]
        self.ws = info['ws'].values.astype(float)[0]    # settling velocity [m/s]
        self.HR = info['HR'].values.astype(float)[0]    # Hydraulic radius normalised by flow height
        # self.Frd = Froude_number(self.vx*self.U_ave, self.vy*self.height, self.cx*self.B, self.cy*self.height,self.height)# Froude number
        self.Reynolds = self.U_ave*self.height*self.rhof_ave/0.001 # Reynolds number
        self.d50 = info['d50'].values.astype(float)[0]  # median particle size
        self.isCprofile_available = info['Cprofile'][0]    # Whether the vertical density profile is available or not

    def velocity(self,file_path) -> None:
        # Velocity Profile interpolation
        self.vx, self.vy, self._vx, self._vy, self.h_umax, self.index_hum = velocity_profiler(file_path, self.bh, self.height,islogmode=False)
        self.U_ave = U_ave(self.vx,self.vy,self.height)

    def sedimentWeight(self, cx_mod: np.ndarray, cy_mod: np.ndarray, cx: np.ndarray, cy) -> None:
        # density Profile interpolation (this method is called only by particle-laden flows)
        # sediment concentration: sediment weight per a specific volume in flow [kg/m^3]
        # sedimentConcentration C = rhos * Cv
        ra = self.rhoa
        rs = self.rhos
        self.cx = cx_mod
        self.cy = cy_mod
        self._cx = cx
        self._cy = cy
        # self.cx, self.cy, self._cx, self._cy = fpi.concentration_profile(file_path,self.bh,self.height,self.isConservative,0,height_v) # sediment concentration [g/l]
        # convert sediment concentration [kg/m^3] into volumetric concentration [%]
        self.cvx = self.cx/rs; self._cvx = self._cx/rs; # Volumetric [%*(1/100)]
        # self.cvx = self.cx/(rs-ra); self._cvx = self._cx/(rs-ra); # Volumetric [%*(1/100)]
        self.cvy = self.cy; self._cvy = self._cy
        # convert volumetric concentration [%] into flow density [kg/m^3]
        self.rhofx = (1-self.cvx)*ra+self.cvx*rs; self._rhofx = (1-self._cvx)*ra+self._cvx*rs; # Flow density [g/l]
        self.rhofy = self.cy; self._rhofy = self._cy

        # calc layer-averaged value
        self.C_ave = C_ave(self.cx,self.cy,self.height)  # sediment concentration [kg/m^3]
        self.Cv_ave = C_ave(self.cvx,self.cvy,self.height) # volumetric concentraiton [%]
        self.rhof_ave = C_ave(self.rhofx - ra,self.rhofy,self.height) # flow density [kg/m^3]
        self.rhof_ave = self.rhof_ave + ra
 
        self.m = mu(self.m_type,self.C_ave,rs) # dynamic viscosity
        self.ws = ws(self.g,rs,ra,self.d) # sediment fall velocity [m/s]
    
    def flowDensity(self, cx_mod, cy_mod, cx, cy) -> None:
        # density propile interpolation
        # flow density: total density of flow including fluid and sediments
        ra = self.rhoa # density of ambient water
        self.rhofx = cx_mod
        self.rhofy = cy_mod
        self._rhofx = cx
        self._rhofy = cy
        if self.isConservative == 'Conservative':
            # interpolated
            self.Bx = ( self.rhofx - ra ) / ra
            # Original points
            self._Bx = ( self._rhofx - ra ) / ra
            
        elif self.isConservative == "Non-Conservative":
            rs = self.rhos # density of sediment particle (valid only in the case of sediment-laden flow)
            self.cvx = (self.rhofx-ra)/(rs-ra); self._cvx = (self._rhofx-ra)/(rs-ra)
            self.cvy = self.rhofy; self._cvy = self._rhofy
            self.cx = self.cvx * rs; self._cx = self._cvx * rs; 
            # self.cx = self.cvx * (rs-ra); self._cx = self._cvx * (rs-ra);
            self.cy = self.rhofy; self._cy = self._rhofy
            self.C_ave = C_ave(self.cx,self.cy,self.height)
            self.Cv_ave = C_ave(self.cvx,self.cvy,self.height)
            self.m = mu(self.m_type,self.C_ave,rs)
            self.ws = ws(self.g,rs,ra,self.d)
        self.rhof_ave = C_ave(self.rhofx-ra,self.rhofy,self.height)
        self.rhof_ave = self.rhof_ave + ra
    
    def volumetricConcentration(self, cx_mod, cy_mod, cx, cy) -> None:
        ra = self.rhoa
        rs = self.rhos
        self.cvx = cx_mod
        self.cvy = cy_mod
        self._cvx = cx
        self._cvy = cy
        self.cx = self.cvx*rs; self._cx = self._cvx*rs
        self.cy = self.cvy; self._cy = self._cvy
        self.rhofx = (1-self.cvx)*ra+self.cvx*rs; self._rhofx = (1-self._cvx)*ra+self._cvx*rs
        self.rhofy = self.cvy; self._rhofy = self._cvy
        self.C_ave = C_ave(self.cx,self.cy,self.height)
        self.Cv_ave = C_ave(self.cvx,self.cvy,self.height)
        self.rhof_ave = C_ave(self.rhofx-ra,self.rhofy,self.height)
        self.rhof_ave = self.rhof_ave + ra
        # self.B = self.C_ave * self.R
        self.m = mu(self.m_type,self.C_ave,rs)
        self.ws = ws(self.g,rs,ra,self.d)

    def buoyancy(self, cx_mod, cy_mod, cx, cy) -> None:
        ra = self.rhoa # density of ambient water
        self.Bx = cx_mod
        self._Bx = cx
        self.rhofx = ra * self.Bx + ra
        self._rhofx = ra * self._Bx + ra
        self.rhofy = cy_mod
        self._rhofy = cy
        if self.isConservative == 'Conservative':
            # interpolated
            self.rhofx = ra * self.Bx + ra
            self._rhofx = ra * self._Bx + ra
            
        elif self.isConservative == "Non-Conservative":
            rs = self.rhos # density of sediment particle (valid only in the case of sediment-laden flow)
            self.cvx = (self.rhofx-ra)/(rs-ra); self._cvx = (self._rhofx-ra)/(rs-ra)
            self.cvy = self.rhofy; self._cvy = self._rhofy
            self.cx = self.cvx * rs; self._cx = self._cvx * rs
            self.cy = self.rhofy; self._cy = self._rhofy
            self.C_ave = C_ave(self.cx,self.cy,self.height)
            self.Cv_ave = C_ave(self.cvx,self.cvy,self.height)
            self.m = mu(self.m_type,self.C_ave,rs)
            self.ws = ws(self.g,rs,ra,self.d)
        self.rhof_ave = C_ave(self.rhofx-ra,self.rhofy,self.height)        
        self.rhof_ave = self.rhof_ave + ra

    def fractionalExcessDensity(self, cx_mod, cy_mod, cx, cy) -> None:
        ra = self.rhoa # density of ambient water
        self.rhofx = cx_mod
        self.rhofy = cy_mod
        self._rhofx = cx
        self._rhofy = cy
        self.rhofx = ( self.rhofx*ra ) + ra; self._rhofx = ( self._rhofx*ra ) + ra
        if self.isConservative == 'Conservative':
            # interpolated
            self.Bx = ( self.rhofx - ra ) / ra
            self._Bx = ( self._rhofx - ra ) / ra
            
        elif self.isConservative == "Non-Conservative":
            rs = self.rhos # density of sediment particle (valid only in the case of sediment-laden flow)
            self.cvx = (self.rhofx-ra)/(rs-ra); self._cvx = (self._rhofx-ra)/(rs-ra);
            self.cvy = self.rhofy; self._cvy = self._rhofy;
            self.cx = self.cvx * rs; self._cx = self._cvx * rs;
            self.cy = self.rhofy; self._cy = self._rhofy
            self.C_ave = C_ave(self.cx,self.cy,self.height)
            self.Cv_ave = C_ave(self.cvx,self.cvy,self.height)
            self.m = mu(self.m_type,self.C_ave,rs)
            self.ws = ws(self.g,rs,ra,self.d)
        self.rhof_ave = C_ave(self.rhofx-ra,self.rhofy,self.height)        
        self.rhof_ave = self.rhof_ave + ra

    def set_Cv_ave(self,Cv_ave) -> None:
        ra = self.rhoa
        rs = self.rhos
        self.Cv_ave = Cv_ave
        self.C_ave = Cv_ave*rs
        self.rhof_ave = (1-Cv_ave)*ra + (Cv_ave*rs)
        self.m = mu(self.m_type,self.C_ave,rs)
        self.ws = ws(self.g,rs,ra,self.d)

    def set_C_ave(self,C_ave) -> None:
        ra = self.rhoa
        rs = self.rhos
        self.C_ave = C_ave
        self.Cv_ave = C_ave/rs
        self.rhof_ave = (1-self.Cv_ave)*ra + (self.Cv_ave*rs)
        self.m = mu(self.m_type,self.C_ave,rs)
        self.ws = ws(self.g,rs,ra,self.d)

    def set_Rhof_ave(self,Rhof_ave) -> None:
        ra = self.rhoa
        self.rhof_ave = Rhof_ave
        if self.isConservative == "Conservative":
            self.B = ( Rhof_ave - ra ) / ra
        elif self.isConservative == "Non-Conservative":
            rs = self.rhos
            self.Cv_ave = (Rhof_ave-ra)/(rs-ra);
            self.C_ave = self.Cv_ave*rs
            self.m = mu(self.m_type,self.C_ave,rs)
            self.ws = ws(self.g,rs,ra,self.d)
        

# TC data management
def load_profiles(PROFILE_DIR: str,PARAM_DIR: str,PaperDict: str, ftype: str = '.txt') -> tuple:
    """
    Import compiled data. This method will return the list of runs, list of the names of experiments, and names of runs.

    Parameters
    ------
    PROFILE_DIR: str
        The path of the directory where the vertical flow profile .txt files are kept. The .txt files are created by main.export_results().
        Firstly, run Data_Compilation.ipynb then you should be able to find the directory data/TC/Profile_txt. Also, the folder is specified in the database.py as PROFILE_PATH.
    PARAM_DIR: str
        The path of the directory where the meta data are kept (It is data/TC/Parameter_txt). Also, the folder is specified in the database.py as PARAM_PATH.
    ftype: str = '.txt'
        The file extensnion of the files stored in PROFILE_DIR and PARAM_DIR. It is .txt by default.
    """
    print('loading profiles...')
    run_list = []
    id_exps = []
    id_runs = []
    files = os.listdir(PROFILE_DIR)
    print(str(len(files))+' files are detected')
    # num = len(files)
    # count = 0
    for file in files:
        index = re.search(ftype,file)
        if index:
            run = run_profiler() # initiate
            run.set_id(file[0:-len(ftype)],PaperDict) # set names
            run.load_profiles(PROFILE_DIR + os.sep + file) # velocity & concentration profiles
            run.load_parameter(PARAM_DIR + os.sep + file) # flow parameters
            # store run_profiler() object into a list
            run_list.append(run) 
            id_exps.append(run.exp)
            id_runs.append(run.number)
        # count += 1
    print('loading completed')
    id_exps = list(set(id_exps))
    id_exps.sort()
    return run_list, id_exps, id_runs

def load_ustar(run_list: list, SHEAR_DIR: str, ftype: str = '.txt') -> list:
    """
    Import compiled data. This method will return the list of runs, list of the names of experiments, and names of runs.

    Parameters
    ------
    run_list: list
        A list that stores compiled run_profiler() objects.
    SHEAR_DIR: str
        The path of the directory where the shear velocity information of each run are kept. Shear velocities are estimated from the lower part of velocity profiles. See ShearVelocity.ipynb for the details.
        The file path is specified in database.py (see USTAR_PATH).
    ftype: str = '.txt'
        The file extensnion of the files stored in PROFILE_DIR and PARAM_DIR. It is .txt by default.
    """
    print('loading shear velocity...')
    files = os.listdir(SHEAR_DIR)
    print(str(len(files))+' files are detected')
    for file in files:
        index = re.search(ftype,file)
        if index:
            run_id = file[0:-len(ftype)]
            target_runs = runs_from_number(run_list,[run_id])
            if len(target_runs) > 1:
                print('something wrong with the file name')
            elif len(target_runs) == 0:
                pass
                # print('Given id cannot be found from the path')
            else:
                target_run = target_runs[0]
                ustar_info = pd.read_csv(SHEAR_DIR+os.sep+file)
                target_run.ustar = ustar_info['ustar'].values.astype(float)[0]
    print('loading completed')
    return run_list

def load_fitparams(run_list: list, FITPARAMS_DIR: str, ftype: str = '.csv') -> list:
    """
    Import the fit parameters of concentration functinos with various fit functions.
    It returns a list of run_profiler() objects. If run_list has runs without concentration measurements, they will be omitted from this list.

    Parameters
    ------
    run_list:list
        A list of run_profiler() objects.
    FITPARAMS_DIR: str
        The directory path where the fit parameters are kept.
    ftype: str = '.csv'
        The file extension of the files in the directory, FITPARAMS_DIR.
    """    
    print('loading fit parameters...')
    files = os.listdir(FITPARAMS_DIR)
    run_CV = [run for run in run_list if run.isCprofile_available]
    print('summary\n',str(len(files))+ftype[1:]+' files are detected from the Path\n')
    print('Given list of runs contain '+str(len(run_list))+' runs\n')
    print('of which '+str(len(run_CV))+' runs have voth velocity and conc. profiles\n')
    for run in run_CV:
        file_name = run.id+ftype
        if file_name in files:
            param_dict = {}
            with open(FITPARAMS_DIR + os.sep + file_name, 'r') as file:
                reader = csv.reader(file, delimiter = ',')
                for row in reader:
                    fit_type = row[0] # name of the function for curve fit
                    params_str = row[1] # params as string
                    params_str = params_str[1:-1].split(', ') # split elements and create list
                    params = np.zeros(len(params_str)) # numpy array to store the params
                    for i,p in enumerate(params_str):
                        if p != 'nan':
                            params[i] = float(p) # assign param
                        else:
                            params[i] = np.nan # nan handling
                    param_dict[fit_type] = params
            run.fitparams = param_dict
        else:
            print('Fit params file does not exist for ', run.id) # if file path is not found
    print('loading completed')
    flag = 0
    for run in run_CV:
        if run._cy.values[-1] == 1:
            if run._cy.values[0] == 0:
                run._cx_fit = run._cx
                run._cy_fit = run._cy
            else:
                run._cx_fit = np.append(run.cx[0],run._cx)
                run._cy_fit = np.append(run.cy[0],run._cy)
        else:
            if run._cy.values[0] == 0:
                run._cx_fit = np.append(run._cx,run.cx.values[-1])
                run._cy_fit = np.append(run._cy,run.cy.values[-1])
            else:
                run._cx_fit = np.append(np.append(run.cx[0],run._cx),run.cx.values[-1])
                run._cy_fit = np.append(np.append(run.cy[0],run._cy),run.cy.values[-1])
    print('Upper/lower boundary conditions are added to original profiles')
    print('Fit parameters are loaded successfully')
    return run_CV
    
def runs_id_in_exp(runs: list, exp_name: str) -> list:
    """
    Get list of string of the numbers of runs from the specific experiment.
    
    Parameters
    -----
    runs: list
        A list of the run_profiler() objects from which you want to search the set of runs.
    exp_name: str
        The key of the specific source. See the keys of PaperDict in database.py
    """
    ids = [run.number for run in runs if run.exp == exp_name]
    ids.sort()
    if len(ids) == 0:
        print('#====EXP_HAS_ZERO_RUNS====#')
        print('Filterd by' + exp_name)
        print('#=========================#')
    return ids

def runs_id_in_exps(runs: list, exps_list: list) -> list:
    """
    Get list of string of the ids of the runs from the specified multiple experiments.
    
    Parameters
    -----
    runs: list
        A list of the run_profiler() objects from which you want to search the set of runs.
    exps_list: list
        A list of keys of the experiments. See the keys of PaperDict in database.py
    """
    list_runs = []
    for exp_name in exps_list:
        temp_runs = [[run.number,run.exp] for run in runs if run.exp == exp_name]
        temp_runs.sort()
        list_runs.extend(temp_runs)
    if len(list_runs) == 0:
        print('#-No data has been found--#')
        print('Filterd by' + exp_name)
        print('#-------------------------#')
    return list_runs

def runs_in_exp(runs,exp_name) -> list:
    list_runs = [run for run in runs if run.exp == exp_name]
    if len(list_runs) == 0:
        print('#====EXP_HAS_ZERO_RUNS====#')
        print('Filterd by' + exp_name)
        print('#=========================#')
    return list_runs

def runs_without_exp(runs,exp_name) -> list:
    list_runs = [run for run in runs if run.exp != exp_name]
    if len(list_runs) == 0:
        print('#====EXP_HAS_ZERO_RUNS====#')
        print('Filterd by' + exp_name)
        print('#=========================#')
    return list_runs

def runs_in_exps(runs,exps_list) -> list:
    list_runs = []
    for exp_name in exps_list:
        temp_runs = [run for run in runs if run.exp == exp_name]
        list_runs.extend(temp_runs)
    if len(list_runs) == 0:
        print('#====EXP_HAS_ZERO_RUNS====#')
        print('Filterd by' + exp_name)
        print('#=========================#')
    return list_runs

def runs_from_number(runs,ids) -> list:
    targets = [run for run in runs if run.id in ids ]
    if len(targets) == 0:
        pass
        # print('Nothing found from the given ids')
    return targets

def runs_non_conservative(runs) -> list:
    runs_NC = [run for run in runs if run.isConservative == False]
    return runs_NC

def runs_conservative(runs):
    runs_NC = [run for run in runs if run.isConservative == True]
    return runs_NC

def runs_isCprofile_available(runs,isCprofile):
    runs_VC = [run for run in runs if run.isCprofile_available == isCprofile]
    return runs_VC

# == Calc method for each flow parameter == #

def mu(mu_type:str,c_ave:float,rhos:float):
    """
    Dynamic viscosity. Currently it is set as the constant (0.001)
    """
    # C_vol = 100*c_ave/rhos
    ans = 0.001
    return ans

def velocity_maximum(Uave: float, varray: list):
    """
    Velocity maximum value.
    
    Parameter
    ------
    Uave: float
        Depth-averaged flow velocity [m/s]
    varray: list
        Normalised velocity profile (u(z)/Uave).
    """
    umax = Uave * max(varray)
    return umax

def U_ave(xv: list, yv: list, height: float):
    """
    Depth-averaged flow velocity
    
    Parameter
    ------
    xv: list
        array of streamwise flow velocity [m/s].
    yv: list
        array of height [m] of each velocity value in xv.
    height: float
        flow depth [m]
    """
    S = trapz(xv,yv)
    return S/height # meter

def U_ave_atUmax(xv: list, yv: list, h_um: float):
    """
    Depth-averaged flow velocity lower than the velocity-maximum height.

    Parameter
    -----
    xv: list
        array of streamwise flow velocity [m/s].
    yv: list
        array of height [m] of each velocity value in xv.
    h_um: float
        velocity-maximum height [m].
    """
    xv_um = xv[0:np.argmax(xv)+1]
    yv_um = yv[0:np.argmax(xv)+1]
    S = trapz(xv_um,yv_um)
    u_ave = S/h_um
    return S/h_um # meter

def U_ave_overUmax(xv,yv,h):
    xv_um = xv[np.argmax(xv):]
    yv_um = yv[np.argmax(xv):]
    S = trapz(xv_um,yv_um)
    hvm = np.array(yv_um)[0]
    u_ave = S / (h-hvm)
    return u_ave # meter

def C_ave(xc,yc,height):
    # print(xc)
    S = trapz(xc,yc)
    c_ave = S/height
    return c_ave

def C_ave_atUmax(xc,yc,index_hvm):
    xc_um = xc[0:index_hvm+1]
    yc_um = yc[0:index_hvm+1]
    height = np.array(yc_um)[-1]
    S = trapz(xc_um,yc_um)
    c_ave = S/height
    return c_ave

def C_ave_overUmax(xc,yc,index_hvm):
    xc_um = xc[index_hvm:]
    yc_um = yc[index_hvm:]
    height = np.array(yc_um)[-1]
    S = trapz(xc_um,yc_um)
    c_ave = S/(height - yc[index_hvm])
    return c_ave

def C_ave_ET(xc,yc,xv,U_et,h_et):
    S = trapz(xc*xv,yc)
    c_ave = S / (U_et * h_et)
    return c_ave

def calc_ustar(vx,vy,start_shear,end_shear,log=True):
    Karman = 0.4
    low = np.max([0,start_shear])
    high = np.min([len(vx),end_shear])
    vxm = list(vx[low:high])
    vym = list(vy[low:high])
    starlist = []
    for i in range(len(vxm)-1):
        grad = ( vxm[i+1] - vxm[i] ) / ( vym[i+1] - vym[i] )
        star = grad * Karman * ( ( vym[i+1] + vym[i] ) / 2 )
        starlist.append(star)
    ustar = np.mean(np.array(starlist))
    # cD = np.power( ustar/run.U_ave , 2 )
    if log:
        print('ustar: ',ustar)
        # print('cD: ',cD)
    return ustar

def calc_ustar_inflection(vx,vy,log=False,returnall=False,shearrange=[1,1]):
    vmi = np.argmax(vx) # velocity maximum index
    dudz = np.gradient(vx[1:vmi],np.log10(vy[1:vmi]),edge_order=2) # velocity gradient
    d2udz = np.gradient(dudz,np.log10(vy[1:vmi]),edge_order=2) # second-order differential to seek the inflection points
    inflections = []
    for i in range(len(d2udz)-1):
        if d2udz[i]*d2udz[i+1] < 0:
            inflections.append(i) # inflection point
    if returnall:
        return calc_ustar(vx,vy,np.min(inflections)+1-shearrange[0],np.min(inflections)+1+shearrange[1],log=log),dudz,d2udz,vy[1:vmi],inflections # calc ustar around the inflection point
    else:
        return calc_ustar(vx,vy,np.min(inflections)+1-shearrange[0],np.min(inflections)+1+shearrange[1],log=log)

def ws(g,rho_s,rho_a,d):
    mu = 0.001
    ans = 0
    if d < 100*np.power(0.1,6) :
        ans = (g*(rho_s-rho_a)*d*d)/(18*mu)
    else:
        nu = 1.004* np.power(0.1,6)
        Ds = np.power(g * (rho_s - rho_a)/(rho_a*np.power(nu,2)),1/3)*d
        ans = (nu/d) * ( np.sqrt( ( np.power(10.36,2) + 1.049 * np.power(Ds,3)) ) - 10.36 )
    return ans

def ws_hum(g,rho_s,rho_a,d,mu,c_ave_hum):
    ans = 0
    if d < 100:
        ans = (g*(rho_s-rho_a)*d*d)/(18*mu)
    else:
        nu = mu / ( c_ave_hum + rho_a ) 
        Ds = np.power(g * ((rho_s - rho_a)/rho_a)*np.power(nu,2),1/3)*d
        ans = (nu/d) * ( np.sqrt( ( np.power(10.36,2) + 1.049 * np.power(Ds,3)) ) - 10.36 )
    return ans

def ew(run: run_profiler) -> float:
    """
    The function of water entrainment rate. 
    The formula is from Parker et al. (1987)
    see https://doi.org/10.1080/00221688709499292 for the detail.
    """
    return 0.0750 /np.sqrt( 1 + 718* np.power(run.height*run.g*run.B/np.power(run.U_ave,2),2.4 ) )

def ew_direct(h,g,B,U):
        return 0.0750 /np.sqrt( 1 + 718* np.power(h*g*B/np.power(U,2),2.4 ) )
    
def ew_pdc(rio):
    return 0.21 *np.power(1.0/rio, 1.1)

def rio(gprime,h,theta,U):
    return gprime*h*np.cos(theta)/np.power(U,2)

def Froude_number(xv,yv,xc,yc,h,g=9.8):
    Frd = U_ave(xv,yv,h)/np.sqrt(C_ave(xc,yc,h)*g*h)
    return Frd

def shields_number(ustar,R,g,D):
    tau_star = np.power(ustar,2)/(R*g*D)
    return tau_star

def particleReynolds(R,g,D,nu):
    Rep = np.sqrt(R*g*D)*D/nu
    return Rep

def critical_shields(Rep): # Neil(1968) Parker et al. (2003)
    repm06 = np.power(1.0/Rep,0.6)
    repm05 = np.power(1.0/Rep,0.5)
    tauc = 0.5 * ( 0.22*repm05 + 0.06*np.power(0.1,7.7*repm06))
    return tauc

# Critical shields number from Guo (2020)
def Dstar(D,R,g,nu):
    return D * np.power(R*g/np.power(nu,2),1/3)

def critical_shields_guo(Repc,Dstar):
    return np.power(Repc,2)/np.power(Dstar,3)

def Repc(Dstar):
    coeffs_set = np.array([ np.ones(len(Dstar))*1.0,np.ones(len(Dstar))*(195.0/7.0),(162.0/7.0)-(np.power(Dstar,3)/18.0),-np.power(Dstar,3)*(11/42),-np.power(Dstar,3)*(81/14)])
    ans_list = []
    for i in range(len(Dstar)):
        coeffs = coeffs_set[:,i]
        roots = np.roots(coeffs)
        real_roots = roots[np.isreal(roots)]
        if np.isreal(max(real_roots)):
            ans = np.real(max(real_roots))
        ans_list.append(ans)
    return ans_list

# Structure function
def Structure_velocity(xv,yv,u_ave):
    # u_ave = U_ave(xv,yv) # Layer averaged velocity
    height = max(yv) # flow height (defined as the height where the velocity becomes zero)
    S = trapz(xv*xv,yv)
    return S / (u_ave*u_ave*height)

def Structure_concentration(num,xv,yv,xc,yc,u_ave,rhof_ave,height,rhoa):

    S = trapz(xv*((xc-rhoa)/rhoa),yv)
    return S / (u_ave*((rhof_ave-rhoa)/rhoa)*height)

def Structure_Buoyancy(h,xc,yc,B_ave,rho_a):
    ra = rho_a
    height = h
    By = (xc - rho_a)/rho_a
    for item in (xc-rho_a):
        if item <0:
            # pass
            print("HOGEEEEEEEEEE")
            print("xc: "+str(item+rho_a)+" rhoa: "+str(rho_a))
    S = trapz(By*yc,yc)
    B = S # meter
    return 2*B/(B_ave*h*h)

def tau_star_c(tau_c,rhos,rhoa,d50):
    denom = (rhos - rhoa) * d50
    return tau_c / denom

def convertCw2Cv(cw,rhos):
    cv = cw/rhos
    return cv

def profile_interpolate(x,y,plen):
    from scipy import interpolate
    f = interpolate.PchipInterpolator(y,x)
    y_pchip = np.linspace(min(y),max(y),num=plen)
    x_pchip = f(y_pchip)
    return x_pchip,y_pchip

def interpolatedFunc(x,y):
    from scipy import interpolate
    return interpolate.PchipInterpolator(x,y)

def Uh_ET(xv,yv):
    S_1 = trapz(xv,yv)
    S_2 = trapz(xv*xv,yv)
    u_ave = S_2/S_1
    h_et = ( S_1 * S_1 ) / S_2
    return u_ave,h_et # meter

def C_ET(xc,yc,xv,U_et,h_et):
    S = trapz(xc*xv,yc)
    c_ave = S / (U_et * h_et)
    return c_ave

def velocity_list(FOLDA_DIR,format='.txt'):
    file_list = []
    files = os.listdir(FOLDA_DIR)
    for file in files:
        index = re.search('velocity'+format,file)
        if index:
            file_list.append(file[0:-1*len(format)])
    return file_list

# file list of concentration profile
def concentration_list(FOLDA_DIR,format='.txt'):
    file_list = []
    files = os.listdir(FOLDA_DIR)
    for file in files:
        index = re.search('concentration'+format,file)
        if index:
            file_list.append(file[0:-1*len(format)])
    return file_list

# file list of concentration profile
def stress_list(FOLDA_DIR,format='.txt'):
    file_list = []
    files = os.listdir(FOLDA_DIR)
    for file in files:
        index = re.search('stress'+format,file)
        if index:
            file_list.append(file[0:-1*len(format)])
    return file_list

# file list of concentration profile
def turb_list(FOLDA_DIR,format='.txt'):
    file_list = []
    files = os.listdir(FOLDA_DIR)
    for file in files:
        index = re.search('turb'+format,file)
        if index:
            file_list.append(file[0:-1*len(format)])
    return file_list

def get_run_from_runs(RUN_NAME,RUN_LIST):
    file_list = []
    for run in RUN_LIST:
        index = re.search(RUN_NAME,run)

#  Interpolation and Extrapolation

def initial_set_velocity(x,y,bh,iscm=True,islogmode=True):
    if iscm:
        x_re = np.array(x) / 100 # meter/second
        y_re = ( np.array(y) - bh ) / 100 # meter / second
    else:
        x_re = np.array(x) # meter/second
        y_re = ( np.array(y) - bh )  # meter / second
    # Delete negatie values
    er_index = []
    for i,val in enumerate(x_re):
        if val < 0.0:
            if islogmode:
                print("#=====#")
                print("V_init_error: negative x_value in original dataset")
                print("#=====#")
            er_index.append(i)
        if y_re[i] < 0.0:
            if islogmode:
                print("#=====#")
                print("V_init_error: negative y_value in original dataset")
                print("#=====#")
            er_index.append(i)
            # y_re[i] = 0
    x_re = np.delete(x_re,er_index)
    y_re = np.delete(y_re,er_index)
    # insert boundary condition at bed (0)
    # x_re = np.insert(x,0,0.0)
    # y_re = np.insert(y,0,0.0)
    if y_re[0] != 0.0:
        x_re = np.append([0.0],x_re)
        y_re = np.append([0.0],y_re)
    # print(x_re)
    # print("00000")
    return x_re, y_re, y_re[-1]

def velocity_extrapolate(x,y,islogmode=True):
    ym = y[-1]
    ym_ind = len(y)
    vm_ind = np.argmax(x)
    fit_ind = int( (ym_ind + vm_ind) / 2 )
    if ym_ind <=3:
        fit_ind = ym_ind
#     fit_ind = np.argmax(x)
    def func_fit(x,a,b,c):
        # return a * np.power(x,5) + b * np.power(x,4) + c * np.power(x,3) + d * np.power(x,2) + e * np.power(x,1) + f
        return np.power(a,2) * np.power(x,2) + b * x + c
    param, cov = curve_fit(func_fit,x[fit_ind:],y[fit_ind:],maxfev=100000)
    x_poly = np.linspace(0,np.max(x)*2,num=300)
    # y_poly = func_fit(x_poly, param[0],param[1],param[2],param[3],param[4])
    y_poly = func_fit(x_poly, param[0],param[1],param[2])
    if y_poly[0] <= ym:
        if islogmode:
            print("Vel: failed Quadratic Extrapolation. switched to linear extrapolation")
        grad = ( y[-1] - y[vm_ind] ) / ( x[vm_ind] - x[-1] )
        y_zero = y[-1] + grad * x[-1]
        x_poly = np.append(0,x_poly)
        y_poly = np.append(y_zero,y_poly)
    return x_poly[0], y_poly[0], param

def velocity_profiler(data,bh,iscm=True,height=np.nan,islogmode=True):
    if islogmode:
        print("Velocity_interpolation")

    # initial setting
    x, y, ym_ori = initial_set_velocity(data.x,data.y,bh,iscm=iscm,islogmode=islogmode)
    # interpolate
    x_pchip, y_pchip = profile_interpolate(x,y,30)
    # extrapolate
    if np.isnan(height):
        x_up, y_up, params = velocity_extrapolate(x_pchip,y_pchip,islogmode=islogmode)
    else:
        x_up = 0
        y_up = height
    # create profile
    if x_pchip[-1] != 0:
        if y_up > y_pchip[-1]:
            x_pchip = np.append(x_pchip,x_up)
            y_pchip = np.append(y_pchip,y_up)
        else:
            # Delete negatie values
            er_index = []
            for i,val in enumerate(y_pchip):
                if y_up <= val:
                    er_index.append(i)
            x_pchip = np.delete(x_pchip,er_index)
            y_pchip = np.delete(y_pchip,er_index)
            if len(x_pchip) == 0:
                if islogmode:
                    print('Something terrible happened.')
            x_pchip = np.append(x_pchip,x_up)
            y_pchip = np.append(y_pchip,y_up)
    else:
        x_up = x_pchip[-1]
        y_up = y_pchip[-1]
    x_pchip,y_pchip = profile_interpolate(x_pchip, y_pchip, 500)

    return x_pchip, y_pchip, x, y, x_up, y_up, ym_ori

def initial_set_concentration(x,y,bh,min_value,iscm=True,islogmode=True):
    if iscm:
        x_re = np.array(x) # density
        y_re = ( np.array(y) - bh ) / 100 # meter
    else:
        x_re = np.array(x) # density
        y_re = ( np.array(y) - bh )  # meter

    er_index = []
    # Delete negatie values
    er_index = []
    Cmax_ind = np.argmax(x_re)
    for i,val in enumerate(x_re):
        if val < min_value:
            if islogmode:
                print("#=====#")
                print("C_INIT_ERROR: Some data points are smaller than min_value")
                print("#=====#")
            er_index.append(i)
        if y_re[i] < 0:
            if islogmode:
                print("#=====#")
                print("V_INIT_ERROR: negative y_value in original dataset")
                print("#=====#")
            er_index.append(i)
        if x_re[i] < x_re[Cmax_ind] and i < Cmax_ind:
            er_index.append(i)
    x_re = np.delete(x_re,er_index)
    y_re = np.delete(y_re,er_index)
    return x_re, y_re, y_re[-1]

def concentration_upper_polynomial(x,y,min_value,islogmode=True):
    ym = y[-1]
    ylen = len(y)
    fit_ind = int(ylen/2)
    if ylen <=4:
        fit_ind = ylen
    def func_fit(x,a,b,c):
        return np.power(a,2) * np.power(x,2) + b * x + c

    param,cov = curve_fit(func_fit,x[fit_ind:]-min_value,y[fit_ind:],maxfev=10000000)
    x_poly = np.linspace(x[-3]-min_value, 0, num = 30)
    y_poly = func_fit(x_poly,param[0],param[1],param[2])
    # adjust y values
    er_index = []
    for i,val in enumerate(y_poly):
        if val == np.inf:
            er_index.append(i)
    x_poly = np.delete(x_poly,er_index)
    y_poly = np.delete(y_poly,er_index)
    x_poly = x_poly + min_value
    # Exception handling
    if x_poly[-1] != min_value:
        grad = ( y_poly[-1] - y_poly[-2] ) / ( x_poly[-2] - x_poly[-1] )
        y_zero = y_poly[-1] + grad * x_poly[-1]
        x_poly = np.append(x_poly,min_value)
        y_poly = np.append(y_poly,y_zero)

    if y_poly[-1] <= ym:
        if islogmode:
            print("Conc: failed Extrapolation")
        grad = ( y[-1] - y[-2] ) / ( x[-2] - x[-1] )
        y_zero = y[-1] + grad * x[-1]
        x_poly = np.append(x_poly,min_value)
        y_poly = np.append(y_poly,y_zero)
    
    return x_poly[-1],y_poly[-1],x_poly,y_poly,param

def concentration_lower_polynomial(x_ori,y_ori,x,y,plen,isConserve):
    x_low = 0
    y_low = 0
    if isConserve == "Conservative":
        x_low = x[0]
        y_low = 0
        return x_low, y_low, x_low, y_low
    else:
        xlen = len(y)
        xup = int(xlen/3)
        y_poly = np.linspace(0,min(y),num=30)
        x_poly = np.poly1d( np.polyfit( y[0:xup], x[0:xup], 3 ) )(y_poly)
    x_bed = x_ori[1] - ( y_ori[1] * (x_ori[0]-x_ori[1])/(y_ori[0]-y_ori[1]) )
    return x_bed, y_poly[0], x_poly, y_poly
    
def concentration_profiler(data,bh,min_value,isConserve,height=np.nan,iscm=True,islogmode=True):
    if islogmode:
        print("Concentration_interpolation")
    # initial setting
    x, y, ym_ori = initial_set_concentration( data.x, data.y, bh, min_value,iscm=iscm,islogmode=islogmode)
    # interpolate Pchip
    x_pchip, y_pchip = profile_interpolate(x,y,300)
    # Upper polynomial extrapolation
    if np.isnan(height):
        x_up, y_up, x_poly_up, y_poly_up, params = concentration_upper_polynomial(x_pchip,y_pchip,min_value,islogmode=islogmode)
    else:
        x_up = min_value
        y_up = height
    # Lower polynomial extrapolation
    x_low, y_low, x_poly_lw, y_poly_lw = concentration_lower_polynomial(x, y, x_pchip, y_pchip, 500, isConserve)
    # create profile
    if y[0] != 0:
        if x[0] > x_low: # If extrapolated C_zero is lower than the lowest original measurement
            x_pr = np.append(x[0],x)
            y_pr = np.append(y_low,y)
        else:
            x_pr = np.append(x_low,x)
            y_pr = np.append(y_low,y)
    else:
        x_pr = x
        y_pr = y
    if x_pr[-1] != min_value:
        if y_up > y_pr[-1]:
            x_pr = np.append(x_pr,x_up)
            y_pr = np.append(y_pr,y_up)
        else:
            y_up = y_pr[-1]
    else:
        y_up = y_pr[-1]
    x_all, y_all = profile_interpolate(x_pr,y_pr,500)

    return x_all, y_all, x_pr, y_pr, y_up, x, y, ym_ori

def fit_profile(ax,run,func,params,y=np.linspace(0,1,100),color='blue',linestyle='-',label=''):
    import matplotlib.pyplot as plt
    ax.scatter(run._cx_fit,run._cy_fit)
    ax.scatter(run._cx,run._cy)
    ax.plot(max(run.cx)*func(y,*params),y,color=color,linestyle=linestyle,label=label)
    ax.set_title(run.id)

def interceptY_from_two_points(x1,y1,x2,y2):
    if x1 - x2 == 0:
        intercept = np.inf
    else:
        intercept = (x1*y2 - x2*y1) / (x1-x2)
    return intercept

def interceptX_from_two_points(x1,y1,x2,y2):
    if y1 - y2 == 0:
        intercept = np.inf
    else:
        intercept = -1.0 * (x1*y2 - x2*y1) / (y1-y2)
    return intercept

def init_TKE_measurement_points(data):
    init_coords = pd.DataFrame()
    x = data.x.values; y = data.y.values
    x_low = 0; x_high = 0
    y_low = 0; y_high = interceptY_from_two_points(x[-1],y[-1],x[-2],y[-2])
    if y_high < y[-1]:
        y_high = interceptY_from_two_points(x[-1],y[-1],x[-3],y[-3])
    x_init = np.concatenate(([x_low],x,[x_high]))
    y_init = np.concatenate(([y_low],y,[y_high]))
    init_coords["x"] = x_init
    init_coords["y"] = y_init
    return init_coords


def TKE_profiler(data):
    profile = pd.DataFrame()
    profile["x"], profile["y"] = profile_interpolate(data.x,data.y,500)
    return profile

def init_STRESS_measurement_points(data):
    init_coords = pd.DataFrame()
    x = data.x.values; y = data.y.values
    x_low = interceptX_from_two_points(x[0],y[0],x[1],y[1]) ; x_high = 0
    y_low = 0; y_high = interceptY_from_two_points(x[-1],y[-1],x[-2],y[-2])
    if y_high < y[-1]:
        y_high = interceptY_from_two_points(x[-1],y[-1],x[-3],y[-3])
    x_init = np.concatenate(([x_low],x,[x_high]))
    y_init = np.concatenate(([y_low],y,[y_high]))
    init_coords["x"] = x_init
    init_coords["y"] = y_init
    return init_coords

def STRESS_profiler(data):
    profile = pd.DataFrame()
    profile["x"], profile["y"] = profile_interpolate(data.x,data.y,500)
    return profile

def re_interp(data,plen,relen):
    re = pd.DataFrame()
    profile = pd.DataFrame()
    re["x"], re["y"] = profile_interpolate(data.x,data.y,relen)
    profile["x"], profile["y"] = profile_interpolate(re["x"],re["y"],plen)
    return profile

def dx_dy(x,y,kind='pchip'):
    # NOTE y must be arithmatic sequence
    gradient = pd.DataFrame()
    
    dy = (np.absolute(y[0]-y[1]))/2.0
    if kind == 'linear':
        f = interpolate.interp1d(y,x)
    else:
        f = interpolate.PchipInterpolator(y,x)
    grad = []
    for i, val in enumerate(y):
        if i == 0:
            grad.append( ( f(y[i]+dy) - x[i] )/dy )
        elif i == len(y)-1:
            grad.append( ( x[i] - f(y[i]-dy) )/dy )
        else:
            grad.append( ( f(y[i]+dy) - f(y[i]-dy) )/(2*dy) )
    gradient['y'] = y
    gradient['x'] = np.array(grad)
    return gradient

def height_for_trapz(ya,yb):
    height = min(ya,yb)
    return height

def profile_for_trapz(data,h_trapz,plen):
    pt = pd.DataFrame()
    ft = interpolate.PchipInterpolator(data.y,data.x)
    pt["y"] = np.linspace(0,h_trapz,plen)
    pt["x"] = ft(pt["y"])
    # print(pt["y"])
    return pt
