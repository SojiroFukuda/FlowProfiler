#======= meta data of each experiments ==================
TC_info = "data/TC/meta/metadata.csv"
PDC_info = 'data/PELE/PELE_info.csv'
Coord_dir = "Coord" # Folder where the coordinates of original profiles locate
TC_DIR = "data/TC" # Directory where the data related to TC locates
TC_Paper_DIR = "data/TC/Paper" # Directory where info of each TC source locates
PDC_DIR = "data/PELE" # Folder where the data of Pyroclastic Density Currents locates
# PATH of compiled Database of Empirical Source #
PROFILE_PATH = 'data/TC/Profile_txt' # Interpolated/extrapolated Profile coordinates
PARAM_PATH = 'data/TC/Parameter_txt' # Flow Parameters

PROFILEPDC_PATH = 'data/PELE/Profile_txt' # Interpolated/extrapolated Profile coordinates
PARAMPDC_PATH = 'data/PELE/Parameter_txt' # Flow Parameters
USTAR_PATH = 'data/TC/ShearVelocity' # Shear velocities
USTARPDC_PATH = 'data/PELE/ShearVelocity' # Shear velocities

FITPARAMS_PATH = 'data/TC/Fitparams' # Curve fit parameters
EXPORT_DIR = 'Export' # Export directory
##################################
# def get_expInfo(): 
#     return exp_info
def get_metaPath_TCs():
    return (PROFILE_PATH,PARAM_PATH,USTAR_PATH,FITPARAMS_PATH)

def get_metaPath_PDCs():
    return (PROFILEPDC_PATH,PARAMPDC_PATH)

### Dictionary of literature #####
# TCs and Yellow river data
PaperDict = {"Tesakar1969": "Tesaker (1969)", 
"Amy2005": "Amy et al. (2005)", 
"Sequeiros2018":"Sequeiros et al. (2018)", 
'Altinakar1988':'Altinakar (1988)',
'Sequeiros2010bedload':'Sequeiros et al. (2010b)',
'Sequeiros2010mobilebed':'Sequeiros et al. (2010a)',
'Eggenhuisen2019':'Eggenhuisen et al. (2019)',
'Michon1955': 'Michon et al. (1955)',
'Cartigny2013': 'Cartigny et al. (2013)',
'Leeuw2017': 'Leeuw et al. (2017)',
'Leeuw2018': 'Leeuw et al. (2018)',
'Kelly2019': 'Kelly et al. (2019)',
'Koller2019': 'Koller et al. (2019)',
'Varjavand2015': 'Varjavand et al. (2015)',
'vanMaren': 'Wan & Wang (1994)',
'Hermidas2018': 'Hermidas et al. (2018)',
'Fedele2016': 'Fedele et al. (2016)',
'IslamImran2010': 'Islam & Imran (2010)',
'Pohl2020': 'Pohl et al. (2020)',
'Simmons2020': 'Simmons et al. (2020)',
'DeepExp2021': 'This study (Quartz)',
'DeepExp2021MdV': 'This study (Kaolinite)',
'PeakallTemp': 'Peakall (saline)',
'Packman2004': 'Packman & Jerolmack (2004)',
'Garcia1993': 'Garcia (1993)',
'Farizan2019': 'Farizan et al. (2019)',
'Fluvial': 'Wan & Wang (1994)'
} 

# PDCs
PaperDictPDC = {
'BroschLube2020':'Brosch & Lube (2020)',
'BreardLube2017':'Breard & Lube (2017)'
}

# TCs only
PaperDictTC = {"Tesakar1969": "Tesaker (1969)", 
"Amy2005": "Amy et al. (2005)", 
"Sequeiros2018":"Sequeiros et al. (2018)", 
'Altinakar1988':'Altinakar (1988)',
'Sequeiros2010bedload':'Sequeiros et al. (2010b)',
'Sequeiros2010mobilebed':'Sequeiros et al. (2010a)',
'Eggenhuisen2019':'Eggenhuisen et al. (2019)',
'Michon1955': 'Michon et al. (1955)',
'Cartigny2013': 'Cartigny et al. (2013)',
'Leeuw2017': 'Leeuw et al. (2017)',
'Leeuw2018': 'Leeuw et al. (2018)',
'Kelly2019': 'Kelly et al. (2019)',
'Koller2019': 'Koller et al. (2019)',
'Varjavand2015': 'Varjavand et al. (2015)',
'Hermidas2018': 'Hermidas et al. (2018)',
'Fedele2016': 'Fedele et al. (2016)',
'IslamImran2010': 'Islam & Imran (2010)',
'Pohl2020': 'Pohl et al. (2020)',
'Simmons2020': 'Simmons et al. (2020)',
'DeepExp2021': 'This study (Quartz)',
'DeepExp2021MdV': 'This study (Kaolinite)',
'Packman2004': 'Packman & Jerolmack (2004)',
'Garcia1993': 'Garcia (1993)',
'Farizan2019': 'Farizan et al. (2019)',
} 

##################################
            
def get_PaperDict():
    return PaperDict
def get_PaperDictPDCs():
    return PaperDictPDC
def get_PaperDictTCs():
    return PaperDictTC
def get_PaperRef(key):
    return PaperDict[key]

### Data Modification #############
import numpy as np
import flowprofiler as fp

def ws(g,rho_s,rho_a,d,mu,c_ave):
    ans = 0
    if d < 100*np.power(0.1,6) :
        ans = (g*(rho_s-rho_a)*d*d)/(18*mu)
    else:
        nu = 1.004* np.power(0.1,6)
        Ds = np.power(g * (rho_s - rho_a)/(rho_a*np.power(nu,2)),1/3)*d
        ans = (nu/d) * ( np.sqrt( ( np.power(10.36,2) + 1.049 * np.power(Ds,3)) ) - 10.36 )
    return ans

def R_squared(observed, predicted, uncertainty=1):
    """ Returns R square measure of goodness of fit for predicted model. """
    weight = 1./uncertainty
    return 1. - (np.var((observed - predicted)*weight) / np.var(observed*weight))
def adjusted_R(x, y, model, popt, unc=1):
    """
    Returns adjusted R squared test for optimal parameters popt calculated
    according to W-MN formula, other forms have different coefficients:
    Wherry/McNemar : (n - 1)/(n - p - 1)
    Wherry : (n - 1)/(n - p)
    Lord : (n + p - 1)/(n - p - 1)
    Stein : (n - 1)/(n - p - 1) * (n - 2)/(n - p - 2) * (n + 1)/n

    """
    # Assuming you have a model with ODR argument order f(beta, x)
    # otherwise if model is of the form f(x, a, b, c..) you could use
    # R = R_squared(y, model(x, *popt), uncertainty=unc)
    R = R_squared(y, model(popt, x), uncertainty=unc)
    n, p = len(y), len(popt)
    coefficient = (n - 1)/(n - p - 1)
    adj = 1 - (1 - R) * coefficient
    return adj, R


def adjust_Czero(run_list):
    # TYPE III: Phi 
    for run in run_list:
        run.Czero_vol = run.Czero # original value of initial conc. in mixing tank (unit varies)
        run.cohesive = False 
        if run.exp == 'Tesakar1969': # g/l
            run.cohesive = True
            run.Czero_vol = (run.rhoa/run.rhos)*run.Czero*np.power(0.1,3)
            run.ws_rev = ws(run.g,run.rhos,run.rhoa,run.d50,0.001,run.Cv_ave)
            run.u3ghw_rev = np.power(run.U_ave,3)/(run.R*run.g*run.height*run.ws_rev)
            run.FP_ustar = run.ustar*run.ustar*run.U_ave/(run.R*run.g*run.height*run.ws)
            run.FP_ustar_rev = run.ustar*run.ustar*run.U_ave/(run.R*run.g*run.height*run.ws_rev)
        elif run.exp == 'Farizan2019':
            run.cohesive = True
            run.Czero_vol = (run.rhoa/run.rhos)*run.Czero*np.power(0.1,3)
        elif run.exp == 'Altinakar1988': # Flow Density
            run.cohesive = False
            run.Czero_vol = (run.Czero - run.rhoa) / (run.rhos - run.rhoa)
        elif run.exp == 'IslamImran2010': # Fractional Excess Density
            run.cohesive = False
            run.Czero_vol = (run.rhoa/(run.rhos-run.rhoa))*run.Czero
        elif run.exp == 'Sequeiros2018': # kg/m^3
            run.cohesive = False
            run.Czero_vol = run.Czero/run.rhos
        elif run.exp == 'Michon1955': # kg/m^3
            run.cohesive = False
            run.Czero_vol = run.Czero/run.rhos
        elif run.exp == 'DeepExp2021MdV':
            run.cohesive = True
        elif run.exp == 'Packman2004':
            if run.d50 < np.power(0.1,6)*5:
                run.cohesive = True
        elif run.exp == 'Varjavand2015':
            run.Czero_vol =  (run.rhoa/run.rhos)*run.Czero*np.power(0.1,3)

### Empirical Phi from Phi_zero ###
ccz_power = 0.93807294
ccz_ratio = np.power(0.1,0.52378806)
Rratio = 3000000
TesakersD50 = 4.987770657346995e-06
##################################    

def reviseFlowParams(run_list):
    for run in run_list:
        # Hydraulic Radius
        if run.exp == 'Leeuw2017':
            h = np.power(0.1,3) * 50 * 62.0/69.0 # [m]
            w = np.power(0.1,3) * 500 * 474.0/223.0 # [m]
            syahen = np.sqrt(np.power(h,2)+ np.power(w/2.0,2))
            junpen = 2.0*syahen
            Area = h * w / 2.0
            run.HR = (Area / junpen) / run.height
        elif run.exp == 'Simmons2020':
            p = 1.0/450
            a = np.sqrt(run.height/p)
            area = 300 * run.height - ( (2/3) * p * np.power(150,3) )
            circum = (1.0/(2.0*p)) * ( p*a*np.sqrt(4*np.power(p*a,2)+1)+0.5*np.log(np.sqrt(4*np.power(p*a,2)+1)+2.0*p*a) )
            rh = area/circum
            areaOverjunpen = run.height*np.sqrt(450*run.height)/(2*np.sqrt(450*run.height+np.power(run.height,2)))
            run.HR = areaOverjunpen/run.height
#             run.HR = rh/run.height
            if np.isnan(run.HR):
                print('HR calc goes wrong: ',run.id)
            elif run.HR <= 0:
                print('negative HR:',run.id)
        # Assign TYPE
        if run.isCprofile_available:
            run.TYPE = 1
        else:
            run.TYPE = 2
        
        if run.exp == 'Simmons2020':
            # run.TYPE = 1
            if 'event01' in run.number:
                run.event = '01'
            elif 'event04' in run.number:
                run.event = '04'
            elif 'event05' in run.number:
                run.event = '05'    
        # # Particle settling velocity and related flow parameters
        run.isEquilibrium = False
        if run.isConservative == False:
            if run.exp == 'Tesakar1969':
                run.d50 = TesakersD50 # revised median size of particle
            if run.isCprofile_available:
                run.ws = ws(run.g,run.rhos,run.rhoa,run.d50,0.001,run.Cv_ave)
                run.u3ghw = run.ustar*run.ustar*run.U_ave/(run.g*run.height*run.ws) 
                run.RouseNum = 6 * run.ws/(run.Karman*run.ustar)
                run.shields = fp.shields_number(run.ustar,run.R,run.g,run.d50)
            else:
                run.Cv_ave = ccz_ratio*np.power(run.Czero_vol,ccz_power)
                run.B = run.R * run.Cv_ave
                run.ws = ws(run.g,run.rhos,run.rhoa,run.d50,0.001,run.Cv_ave)
                run.u3ghw = run.ustar*run.ustar*run.U_ave/(run.g*run.height*run.ws) 
                run.RouseNum = 6 * run.ws/(run.Karman*run.ustar)
                run.shields = fp.shields_number(run.ustar,run.R,run.g,run.d50)
            # Shields check
            run.cD = np.power(run.ustar/run.U_ave, 2)
            run.shields = fp.shields_number(run.ustar,run.R,run.g,run.d50)
            run.Rep = fp.particleReynolds(run.R,run.g,run.d50,np.power(0.1,6))
            dstar = fp.Dstar( run.d50,1.68,9.8,np.power(0.1,6) )
            repc = fp.Repc( [dstar])
            if run.shields < fp.critical_shields_guo( repc,dstar ):
                run.shields_safe = False
            else:
                run.shields_safe = True
            # Power Balance check
            if run.isCprofile_available:
                run.u2ghs_Ri = np.power(run.ustar,2)/(run.g*run.height*run.slope)*(1+fp.ew(run)/np.power(run.ustar/run.U_ave, 2)*(1 + 0.5*(run.height*run.g*run.B/np.power(run.U_ave,2))) )
            else:
                run.u2ghs_Ri = np.power(run.ustar,2)/(run.g*run.height*run.slope)*(1+fp.ew(run)/np.power(run.ustar/run.U_ave, 2)*(1 + 0.5*(run.height*run.g*run.B/np.power(run.U_ave,2))) )
            run.Rforce = run.B/run.u2ghs_Ri
            if run.Rforce > 1.0/Rratio and run.Rforce < Rratio:
                run.power_safe = True
            else:
                run.power_safe = True
            if run.power_safe and run.shields_safe:
                run.isEquilibrium = True
        else:
            run.RouseNum = 0 # Set nought in the case of conservative flows
            
    # distance between the inlet and the measurement location [m]
    islamx = np.array([2.61,3.41,4.21,4.60,5.01,5.41,5.81,6.61,6.97,7.55,7.77,8.43])
    for run in run_list:
        if run.exp == 'Altinakar1988':
            run.L = float(run.number.split('-')[1])/100
        elif run.exp == "Tesakar1969":
            run.L = 6
        elif run.exp == 'Amy2005':
            run.L = 2.5
        elif run.exp == 'Sequeiros2010bedload':
            run.L = 4.5
        elif run.exp == 'Sequeiros2010mobilebed':
            run.L = 4.5
        elif run.exp == 'Eggenhuisen':
            run.L = 2.5
        elif run.exp == 'Michon1955':
            run.L = 38
        elif run.exp == 'Cartigny2013':
            run.L = 2.5
        elif run.exp == 'Leeuw2017':
            run.L = 2.1
        elif run.exp == 'Leeuw2018':
            run.L = 2.1
        elif run.exp == 'Kelly2019':
            run.L = 2
        elif run.exp == 'Koller2019':
            run.L = 14
        elif run.exp == 'Varjavand2015':
            run.L = 2.25
        elif run.exp == 'Hermidas2018': 
            run.L = 2.71
        elif run.exp == 'Fedele2016': 
            run.L = 3
        elif run.exp == 'IslamImran2010': 
            run.L = islamx[int(run.number[4:])-1]
        elif run.exp == 'Pohl2020':
            run.L = 3
        elif run.exp == 'DeepExp2021':
            run.L = 4.7
        elif run.exp == 'DeepExp2021MdV': 
            run.L = 4.7
        elif run.exp == 'Packman2004':
            run.L = 1.4
        elif run.exp == 'Garcia1993': 
            run.L = 2
        elif run.exp == 'Simmons2020':
            run.L = 143000
        elif run.exp == 'Farizan2019':
            if 'D' in run.number:
                run.L = 7.5
            else:
                run.L = 4.0
        else:
            run.L = -1


def reviseFlowParamsPDCs(pdc_list):
    for run in pdc_list:
        if run.exp == 'BroschLube2020':
            run.L = 1.72
        else:
            run.L = 11.72
        run.R = (run.rhos - run.rhoa)/run.rhoa
        run.rhomix = run.rhos*run.Cv_ave + run.rhoa*(1.0-run.Cv_ave)
        run.gprime = (run.rhomix - run.rhoa ) / run.rhoa
        run.theta = np.arctan(run.slope)
        run.rio = fp.rio(run.gprime,run.height,run.theta,run.U_ave)
        run.ew = fp.ew_pdc(run.rio)
        run.cD = (run.ustar**2)/(run.U_ave**2)
        run.Pf_Nf = np.power(run.ustar,2)*run.U_ave/(run.R*run.g*run.height*run.ws)
        run.Pth_Nth = (np.power(run.ustar,2)*run.U_ave + 0.5*run.ew*np.power(run.U_ave,3) )/(run.R*run.g*run.height*(run.ws+0.5*run.U_ave*run.ew))
        run.shields = fp.shields_number(run.ustar,run.R,run.g,run.d50)
        run.Rep = fp.particleReynolds(run.R,run.g,run.d50,np.power(0.1,6))
        dstar = fp.Dstar( run.d50,1.68,9.8,np.power(0.1,6) )
        repc = fp.Repc( [dstar])
        if run.shields < fp.critical_shields_guo( repc,dstar ):
            run.shields_safe = False
        else:
            run.shields_safe = True
        

###### Functions for curve fit ####################

def bias(x,a):
	h = 1-x
	bs = h  / (((1.0/a)-2)*(1-h) + 1)
	return 1-bs

def schlick(x,slope,threshold,epsilon): # Barron 2020
#     s = 8.0**slope
    s = slope
    s = 1+np.exp(slope)
    t = threshold
#     Ps = ((1.0 - s) + s*np.log(s))/np.power(1.0-s,2)
#     PhiR = (2*t-1.0)*Ps + (1-t) 
    h = 1-x
    x_low = h[h<t]
    x_high = h[h>=t]
    ans_low = t*x_low /( x_low + s * (t - x_low) + epsilon )
    ans_high = (1 - t) * (x_high - 1) / (1-x_high-s*(t-x_high) + epsilon ) + 1
    Cz = np.concatenate((ans_high,ans_low), axis = None )
    return Cz

def schlick_c(x,slope,threshold,cmax,epsilon): # Barron 2020
	s = 8.0**slope
	t = threshold
	h = 1-x
	x_low = h[h<t]
	x_high = h[h>=t]
	ans_low = t*x_low /( x_low + s * (t - x_low) + epsilon )
	ans_high = (1 - t) * (x_high - 1) / (1-x_high-s*(t-x_high) + epsilon ) + 1
	profile = np.concatenate((ans_high,ans_low), axis = None )
	return profile*cmax

def schlick_for_velocity(x,a,slope,threshold,epsilon,vmax_height): # Barron 2020
	x_high = (x[x>=vmax_height] - vmax_height )* (1/(1-vmax_height))
	x_low = x[x<vmax_height] * (1/vmax_height)
	y_high = schlick(x_high,slope,threshold,epsilon)
	y_low = bias(x_low,a)
	return np.concatenate((y_low,y_high), axis = None )

def concentration_fit_sim(x,a1,b1):
    return a1*np.exp(b1*x)

def concentration_fit_five(x,a0,a1,a2,a3,a4,a5):
    return a0 + a1*np.power(x,1) + a2*np.power(x,2) + a3*np.power(x,3) + a4*np.power(x,4) + a5*np.power(x,5)

def concentration_fit_sigmoid(x,a,b,cb):
    return -(x*cb)*(1+np.exp(-a+b)) / (x + np.exp(-a * x + b)) + cb

def concentration_fit_sigmoid_rouse(x,a1,a2,a3,a4,a5,a6):
        return ( (a1*x + a2) / ( a3*x + a4*np.exp(a5*x) ) ) + a6
    
#####################################

def get_Schlick_funcs():
    return (bias, schlick, schlick_c, schlick_for_velocity)

def get_fit_funcs():
    return (concentration_fit_sim,concentration_fit_five,concentration_fit_sigmoid,concentration_fit_sigmoid_rouse)
