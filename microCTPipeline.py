"""
Filename: microCTPipeline.py
Description: micro CT image pipeline for bovine trabecular bone
Start date: 28/02/2022
"""
import numpy as np
import SimpleITK as sitk
import porespy as ps
import matplotlib.pyplot as plt
from vedo import volume,show
from vedo.applications import *
import argparse
def boolean_string(skipSeg):
    if skipSeg not in {'False','True'}:
        raise ValueError('Not a valid boolean string')
    return skipSeg =='True'
 
def getInputs():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s','--skipseg',
                        default=False,
                        type=boolean_string, 
                        help='segment image or skip'
                        )
    args = parser.parse_args()
    skipSeg = args.skipseg
    return skipSeg

def signaltonoise(Arr, axis=None, ddof=0):
    #Arr = np.asanyarray(Arr)
    me = Arr.mean(axis)
    #0.33 is desirable
    sd = Arr.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, me/sd)
"""
def stdOfLocalNoise(Arr, axis=None, ddof=0):
    sd = Arr.std(axis=axis,ddof=ddof)
    me = Arr.mean(axis)
    return sd,me
"""
class ImagePipeline():
    def __init__(self):
        self.fL0="./data/Group18/TB11-L0-VOI3.mhd"
        self.fL8="./data/Group18/TB11-L8-VOI3.mhd"
        self.fL10="./data/Group18/TB11-L10-VOI3.mhd"
        self.imageL0=sitk.ReadImage(self.fL0)
        self.imageL8=sitk.ReadImage(self.fL8)
        self.imageL10=sitk.ReadImage(self.fL10)
        self.spacing_=[self.imageL0.GetSpacing(),self.imageL8.GetSpacing(),self.imageL10.GetSpacing()]
        #points to most recent file for displaying
        self.curImgDict={"fL0":self.fL0, "fL8":self.fL8, "fL10":self.fL10}
        self.vfL0=volume.Volume(self.fL0)
        self.vfL8=volume.Volume(self.fL8)
        self.vfL10=volume.Volume(self.fL10)
        self.updatedVols = False
        self.voxelSize=None
        #morphology
        self.fL0Porosity=None
        self.fL0Thickness=None
        self.fL0PoreSize=None
        self.fL0Anisotropy=None
        self.fL8Porosity=None
        self.fL8Thickness=None
        self.fL8PoreSize=None
        self.fL8Anisotropy=None
        self.fL10Porosity=None
        self.fL10Thickness=None
        self.fL10PoreSize=None
        self.fL10Anisotropy=None

    def updateCurImgDict(self):
        """
        writes new image to ./data/Group18/TB11-L*-V013Updated.mhd
        and updates curImgDict keys
        """
        if self.updatedVols is False:
            self.curImgDict["fl0"]="./data/Group18/TB11-L0-VOI3Updated.mhd" 
            self.curImgDict["fl8"]="./data/Group18/TB11-L8-VOI3Updated.mhd"
            self.curImgDict["fl10"]="./data/Group18/TB11-L10-VOI3Updated.mhd"
            self.updatedVols = True
        sitk.WriteImage(self.imageL0,self.curImgDict["fl0"])
        sitk.WriteImage(self.imageL8,self.curImgDict["fl8"])
        sitk.WriteImage(self.imageL10,self.curImgDict["fl10"])
        self.vfL0=volume.Volume(self.curImgDict["fl0"])
        self.vfL8=volume.Volume(self.curImgDict["fl8"])
        self.vfL10=volume.Volume(self.curImgDict["fl10"])
    def preProcess(self):
        """
        Preprocessing of image - Apply three filters then decide between them
        median filter, non-local means filter and gaussian smoothing. Also fix orientation
        """
        #-1 is none, 1 = medianImageFilter, 2 = non-local means filter and 3 = n/a
        best=-1
        noiseFilter = sitk.NoiseImageFilter()
        noiseFilter.SetRadius(3)
        orig = signaltonoise(sitk.GetArrayFromImage(noiseFilter.Execute(self.imageL0)))
        medianfilter = sitk.MedianImageFilter()
        medianfilter.SetRadius(3)
        meanfilter = sitk.MeanImageFilter()
        gaus = sitk.SmoothingRecursiveGaussianImageFilter()
        gaus.SetNormalizeAcrossScale(False)
        gaus.SetSigma(0.0005)
        #median
        L0RatioMedian_=medianfilter.Execute(self.imageL0)
        L0RatioMedian = noiseFilter.Execute(L0RatioMedian_)
        L0RatioMedian = signaltonoise(sitk.GetArrayFromImage(L0RatioMedian_))
        #mean
        L0RatioMean_=meanfilter.Execute(self.imageL0)
        L0RatioMean=noiseFilter.Execute(L0RatioMean_)
        L0RatioMean=signaltonoise(sitk.GetArrayFromImage(L0RatioMean_))
        if L0RatioMean > L0RatioMedian:
            best=2
            L0RatioMedian_=None
        else:
            best=1
            L0RatioMean_=None
        L0RatioGaus_=gaus.Execute(self.imageL0)
        L0RatioGaus=noiseFilter.Execute(L0RatioGaus_)
        L0RatioGaus=signaltonoise(sitk.GetArrayFromImage(L0RatioGaus_))
        if L0RatioMedian_ is None:
            if L0RatioMean > L0RatioGaus:
                best=2
                L0RatioGaus_=None
            else:
                best=3
                L0RatioMean_=None
        else:
            if L0RatioMedian > L0RatioGaus:
                best=1
                L0RatioGaus_=None
            else:
                best=3
                L0RatioMedian_=None
        if best==1:
            self.imageL8=medianfilter.Execute(self.imageL8)
            self.imageL10=medianfilter.Execute(self.imageL10)
            self.imageL0 = L0RatioMedian_
        elif best==2:
            self.imageL8=meanfilter.Execute(self.imageL8)
            self.imageL10=meanfilter.Execute(self.imageL10)
            self.imageL0 = L0RatioMean_
        elif best==3:
            self.imageL8=gaus.Execute(self.imageL8)
            self.imageL10=gaus.Execute(self.imageL10)
            self.imageL0 = L0RatioGaus_
        print("Signal to noise ratios for imageL0: original",orig, " Gauss ", L0RatioGaus, " median ", L0RatioMedian, " mean ", L0RatioMean)

    def displayMeta(self):
        """
        Display image metadata
        """
        print("L0 has voxels of size",self.imageL0.GetSpacing())
        print("L0 has dimensions",self.imageL0.GetSize())
        print("L8 has voxels of size",self.imageL8.GetSpacing())
        print("L8 has dimensions",self.imageL8.GetSize())
        print("L10 has voxels of size",self.imageL10.GetSpacing())
        print("L10 has dimensions",self.imageL10.GetSize())
        self.voxelSize=self.imageL0.GetSpacing()[0]
    def setVoxelSize(self):
        self.voxelSize=0.0065
    def vedoDisplayImage(self):
        print("displaying fL0")
        plt1 = Slicer3DPlotter(self.vfL0)
        plt1.close()
        print("displaying vfL8")
        plt2 = Slicer3DPlotter(self.vfL8)
        plt2.close()
        print("displaying vfL10")
        plt3 =Slicer3DPlotter(self.vfL10)
        plt3.close()

    def psDisplayImage(self):
        """
        Display image
        """
        x = ps.visualization.show_3D(volume(sitk.GetArrayFromImage(self.imageL0)))
        fig,ax = plt.subplots(figsize=[7,7])
        plt.imshow(x)
    
    def histogram(self):
        """
        Histogram
        """
        print("")
    
    def thresholdSegmentation(self):
        """
        perform segmentation based on qualitative threshold results
        """
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(0)
        otsu_filter.SetOutsideValue(1)
        self.imageL0=otsu_filter.Execute(self.imageL0)
        self.imageL8=otsu_filter.Execute(self.imageL8)
        self.imageL10=otsu_filter.Execute(self.imageL10)
    
    def morphologyVals(self):
        self.porosity()
        print("Porosity fL0", self.fL0Porosity, "fL8 ", self.fL8Porosity, "fL10", self.fL10Porosity)
        self.poreSize()
        print("poreSize fL0", [np.mean(self.fL0PoreSize),np.std(self.fL0PoreSize)], "fL8 ", [np.mean(self.fL8PoreSize),np.std(self.fL8PoreSize)], "fL10", [np.mean(self.fL10PoreSize),np.std(self.fL10PoreSize)])
        self.thickness()
        print("thickness fL0", self.fL0Thickness, "fL8 ", self.fL8Thickness, "fL10", self.fL10Thickness)
 
    def thickness(self):
        fL0Thickness_ = ps.filters.local_thickness(sitk.GetArrayFromImage(self.imageL0).astype(np.uint8).T) #* self.spacing_[0]
        self.fL0Thickness=[np.mean(fL0Thickness_),np.std(fL0Thickness_)]
        fL8Thickness_ = ps.filters.local_thickness(sitk.GetArrayFromImage(self.imageL8).astype(np.uint8).T) #* self.spacing_[1]
        self.fL8Thickness=[np.mean(fL8Thickness_),np.std(fL8Thickness_)]
        fL10Thickness_ = ps.filters.local_thickness(sitk.GetArrayFromImage(self.imageL10).astype(np.uint8).T) #* self.spacing_[2]
        self.fL10Thickness=[np.mean(fL10Thickness_),np.std(fL10Thickness_)]
    def porosity(self):
        self.fL0Porosity=ps.metrics.porosity(sitk.GetArrayFromImage(self.imageL0).astype(np.uint8).T)
        self.fL8Porosity=ps.metrics.porosity(sitk.GetArrayFromImage(self.imageL8).astype(np.uint8).T)
        self.fL10Porosity=ps.metrics.porosity(sitk.GetArrayFromImage(self.imageL10).astype(np.uint8).T)

    def poreSize(self):
        #pore diameter - 0.15 to 0.01 ish micrometers 
        por_ = ps.filters.porosimetry(sitk.GetArrayFromImage(self.imageL0).astype(np.uint8).T)
        fL0PoreSizeDist = ps.metrics.pore_size_distribution(por_,log=False)
        self.fL0PoreSize=fL0PoreSizeDist.R
        por_ = ps.filters.porosimetry(sitk.GetArrayFromImage(self.imageL8).astype(np.uint8).T)
        fL8PoreSizeDist = ps.metrics.pore_size_distribution(por_,log=False)
        self.fL8PoreSize=fL8PoreSizeDist.R
        por_ = ps.filters.porosimetry(sitk.GetArrayFromImage(self.imageL10).astype(np.uint8).T)
        fL10PoreSizeDist = ps.metrics.pore_size_distribution(por_,log=False)
        self.fL10PoreSize=fL10PoreSizeDist.R
    def anisotropy(self):
        import imagej
        ij = imagej.init()
        ijImg = ij.openImage(self.curImgDict["fL0"])
        ij.run(ijImg,"Set Scale...")
        ij.run("org.bonej.wrapperPlugins.AnisotropyWrapper")
         
        print("")
#imageJ particle and anisotropy analysis - see pyimagj
# anisotropy - https://bonej.org/anisotropy
# particle analysis - https://bonej.org/particles
def main():
    print("starting - micro-CT imaging pipeline")
    skipSeg=getInputs()
    print(skipSeg)
    imP = ImagePipeline()
    if skipSeg is False:
        print("Displaying meta data")
        imP.displayMeta()
        imP.vedoDisplayImage()
        input("Press Enter to perform image processing.")
        imP.preProcess()
        imP.updateCurImgDict()
        imP.vedoDisplayImage()
        input("Press Enter to perform segmentation.")
        imP.thresholdSegmentation()
        imP.updateCurImgDict()
        imP.vedoDisplayImage()
    input("Press Enter to assess porosity, thickness, pore size and anisotropy")
    if skipSeg:
        imP.setVoxelSize()
        imP.updateCurImgDict()
    print("Results with voxel size = 1")
    imP.morphologyVals()
    #imP.anisotropy()
    print("ending...")
if __name__=="__main__":
    main()
