import numpy as np
import SimpleITK as sitk
from utils import utils
L0 = "./data/Group18/TB11-L0-VOI3Updated.mhd"
L0stl = "./data/Group18/TB11-L0-VOI3Updated.stl"
L8 = "./data/Group18/TB11-L8-VOI3Updated.mhd"
L8stl = "./data/Group18/TB11-L8-VOI3Updated.stl"
L10 = "./data/Group18/TB11-L10-VOI3Updated.mhd"
L10stl = "./data/Group18/TB11-L10-VOI3Updated.stl"

utils.binaryLabelToSTL(L0,L0stl)
utils.binaryLabelToSTL(L8,L8stl)
utils.binaryLabelToSTL(L10,L10stl)
