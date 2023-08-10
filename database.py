#======= meta data of each experiments ==================
TC_info = "data/TC/meta/metadata.csv"
Coord_dir = "Coord" # Folder where the coordinates of original profiles locate
TC_DIR = "data/TC" # Directory where the data related to TC locates
TC_Paper_DIR = "data/TC/Paper" # Directory where info of each TC source locates
# PATH of compiled Database of Empirical Source #
PROFILE_PATH = 'data/TC/Profile_txt' # Interpolated/extrapolated Profile coordinates
PARAM_PATH = 'data/TC/Parameter_txt' # Flow Parameters
EXPORT_DIR = 'Export' # Export directory
##################################

def get_metaPath_TCs() -> None:
    """Return the paths of the folder as a tuple where the coordinates of vertical flow profiles and other flow paramters are kept.

    Returns:
        tuple: (PROFILE_PATH, PARAM_PATH). PROFILE_PATH is the path of folder where all the txt files of profile coordinates are kept. PARAM_PATH is for the flow parameters.
    """
    return (PROFILE_PATH,PARAM_PATH)

### Dictionary of literature #####
# TCs and Yellow river data
PaperDict = {
'DeepExp2021': 'This study (Quartz)',
'DeepExp2021MdV': 'This study (Kaolinite)',
} 


##################################
            
def get_PaperDict():
    return PaperDict

def get_PaperRef(key):
    return PaperDict[key]