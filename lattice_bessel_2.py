# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# where math libraries are on my personal pc (change it...)
import sys
sys.path.append("/Users/ngc/Desktop/PyCore")
#-------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import my_math_functions as mtm
reload(mtm)
from mayavi import mlab
from scipy.special import factorial, j1, j0
import scipy.signal as sgn
import optics_functions as opt
reload(opt)
from scipy.signal import argrelextrema
from skimage.feature import peak_local_max
#-------------------------------------------------------------------------------
# matplotlib parameters
plt.style.use('seaborn-white')
#plt.xkcd()
plt.rcParams['image.cmap'] = 'gray'
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 18}
plt.rc('font', **font)
#-------------------------------------------------------------------------------
# parameters
f_obj = 20.* 10**3 #[um]
wavelength = .488 #[um]
refr_index = 1.33 # water immersion objective
wavelength = wavelength / refr_index
NA = .29
#-------------------------------------------------------------------------------
def mask_geometry (axial_length, NA, wavelength, focal, refr_index, \
                    pictures = 'n'):
    """It gives the geometry of an annular mask, given some parameters:
            - axial_length = half the distance between excitation prifle zeros; 
            -
            -
    """
    
    # i'm using the correct refractive index for a water immersion objective
    # McCutchen calcs
    a = wavelength / axial_length / refr_index
    R_ext = NA * focal / refr_index
    
    X = np.arcsin(R_ext / focal)
    cosY = a + np.cos(X)
    Y = np.arccos(cosY)
    R_int = np.sin(Y)*focal
	
    print   '\nRequired length = ', 2*axial_length, 'um;\n'\
            '\nR_ext:', round(R_ext, 2), 'um' \
            ' <--->  R_int:', round(R_int, 2), 'um' \
            '\nDelta R = ', round(R_ext-R_int, 2), 'um'
    
    # Inde calcs
    length = wavelength * focal**2 / (R_ext-R_int) \
                            / (R_ext - ((R_ext-R_int)/2)) / refr_index
    print '\nDouble check with Indebeteouw:', round(length, 2), 'um'
    
    # Lateral profile = thickness
    r = np.linspace(-1., 13, 444)
    profile = j1(np.pi * R_ext * r / wavelength / focal * refr_index)/r - \
                j1(np.pi * R_int * r / wavelength / focal * refr_index)/r
    
    if (pictures == 'y'):           
        plt.figure('profile')
        plt.fill_between(r, 0, profile**2/np.amax(profile**2),\
                                                    facecolor = 'deepskyblue')
        plt.xlabel(r'$ r \; [\mu m]$')
        plt.ylabel(r'$Normalized intensity \; [a.u.]$')
        plt.grid()
        plt.show()
    
    return R_ext, R_int
#-------------------------------------------------------------------------------
def evaluate_discrepancy(R_ext, R_int, wavelength, focal, r):
    intensity = 1. 
    ideal_bessel = intensity * 2 * np.pi * R_ext * j0(2 * np.pi * R_ext * r \
                                                        / wavelength / focal)
    intensity = 1. 
    ideal_bessel /= np.amax(ideal_bessel)
    practical_bessel = intensity * wavelength * focal/ 2. * (\
                            j1(np.pi * R_ext * r / wavelength / focal) / r - \
                            j1(np.pi * R_int * r / wavelength / focal) / r)
    corr = mtm.crossCorr(np.abs(practical_bessel), np.abs(ideal_bessel))
    practical_bessel /= np.amax(practical_bessel)
    corr_perfect = mtm.crossCorr(np.abs(ideal_bessel), np.abs(ideal_bessel))
    plt.plot(corr_perfect, 'b')
    print corr_perfect[len(corr_perfect)/2]+1
    print corr[len(corr_perfect)/2]+1
    return ideal_bessel, practical_bessel
#-------------------------------------------------------------------------------
# pupil space
step = 101.# um

x, y = np.arange(-40*10**3, 40.1*10**3, step),\
                                        np.arange(-40*10**3, 40.1*10**3, step)
X, Y = np.meshgrid(x, y)

#R_ext, R_int = 3.6 * 10**3, 3.3 * 10**3 # [um]

R_ext, R_int = mask_geometry(70., NA, .488, 20000., 1.33)

annulus = np.zeros((X.shape))
annulus[X**2+Y**2 <R_ext**2] = 1.
annulus[X**2+Y**2 <R_int**2] = 0.
#-------------------------------------------------------------------------------
# sample space
kx, ky = np.linspace(-1./step/2.*wavelength*f_obj,\
                                        1./step/2.*wavelength*f_obj, len(x)), \
         np.linspace(-1./step/2.*wavelength*f_obj,\
                                        1./step/2.*wavelength*f_obj, len(x))
KX, KY = np.meshgrid(kx, ky)

bessel = np.abs(mtm.FT2(annulus))
#-------------------------------------------------------------------------------
first_min = argrelextrema(bessel[bessel.shape[0]/2,bessel.shape[1]/2+1:],\
                                                                        np.less)
first_min = kx[first_min[0]+len(kx)/2][0]
first_max = argrelextrema(bessel[bessel.shape[0]/2,bessel.shape[1]/2+1:],\
                                                                    np.greater)
first_max = kx[first_max[0][0]+len(kx)/2-1] # qui?!

print '\nfirst min:', first_min, 'um'
print '\nfirst max: ', first_max, 'um' 
print '\n\'Total\' central peak: ', first_min * 2., 'um'
#-------------------------------------------------------------------------------
# qui avrei dovuto usare entrambe le direzioni, così è come barare
seed = first_max *2
reciprocal_seed = wavelength * f_obj / seed

print R_ext, R_int, reciprocal_seed

stripes = np.zeros((annulus.shape))
#stripes[:, stripes.shape[0]/2] = 1.
for i in range (len(y)):
    for j in range (0, 20):
        if (np.abs(np.abs(y[i]) - j*reciprocal_seed) <= step/2.):
            stripes[:,i] = 1.
# #-----------------------------------------------------------------------------
seed = first_max * 2.
reciprocal_seed = wavelength * f_obj / seed

print R_ext, R_int, reciprocal_seed

stripes2 = np.zeros((annulus.shape))
#stripes2[:, stripes2.shape[0]/2] = 1.
for i in range (len(y)):
    for j in range (1, 30):
        if (np.abs(np.abs(y[i]) - j*reciprocal_seed) <= step/2.):
            stripes2[:,i] = 1.
#-------------------------------------------------------------------------------
seed = first_max  * 4.
reciprocal_seed = wavelength * f_obj / seed

#reciprocal_seed -= 20

print R_ext, R_int, reciprocal_seed

stripes3 = np.zeros((annulus.shape))
#stripes3[:, stripes3.shape[0]/2] = 1.
for i in range (len(y)):
    for j in range (0, 30):
        if (np.abs(np.abs(y[i]) - j*reciprocal_seed) <= step/2.):
            stripes3[:,i] = 1.
#-------------------------------------------------------------------------------
intensity = np.exp(-np.pi * (X**2)/100./wavelength**2/f_obj**2)*\
                np.exp(-np.pi * (Y**2)/wavelength**2/f_obj**2)

profile = np.abs(mtm.FT2(annulus*stripes*intensity))**2
profile2 = np.abs(mtm.FT2(annulus*stripes2*intensity))**2
profile3 = np.abs(mtm.FT2(annulus*stripes3*intensity))**2
recap = plt.figure('profile')
recap.add_subplot(231)
plt.title('2r')
plt.imshow(profile, cmap = 'jet', interpolation = 'none', \
                                            extent = (\
                                            -1./step/2.*wavelength*f_obj,\
                                            1./step/2.*wavelength*f_obj,\
                                            -1./step/2.*wavelength*f_obj,\
                                            1./step/2.*wavelength*f_obj))
plt.imshow(bessel, alpha = .1, interpolation = 'none', \
                                            extent = (\
                                            -1./step/2.*wavelength*f_obj,\
                                            1./step/2.*wavelength*f_obj,\
                                            -1./step/2.*wavelength*f_obj,\
                                            1./step/2.*wavelength*f_obj))
plt.xlabel(r'$\mu m$'), plt.ylabel(r'$\mu m$')
recap.add_subplot(232)
plt.title('4r')
plt.imshow(profile2,cmap = 'jet',  interpolation = 'none', \
                                            extent = (\
                                            -1./step/2.*wavelength*f_obj,\
                                            1./step/2.*wavelength*f_obj,\
                                            -1./step/2.*wavelength*f_obj,\
                                            1./step/2.*wavelength*f_obj))
plt.imshow(bessel, alpha = .1, interpolation = 'none', \
                                            extent = (\
                                            -1./step/2.*wavelength*f_obj,\
                                            1./step/2.*wavelength*f_obj,\
                                            -1./step/2.*wavelength*f_obj,\
                                            1./step/2.*wavelength*f_obj))
plt.xlabel(r'$\mu m$'), plt.ylabel(r'$\mu m$')
recap.add_subplot(233)
plt.title('6r')
plt.imshow(profile3, cmap = 'jet', interpolation = 'none', \
                                            extent = (\
                                            -1./step/2.*wavelength*f_obj,\
                                            1./step/2.*wavelength*f_obj,\
                                            -1./step/2.*wavelength*f_obj,\
                                            1./step/2.*wavelength*f_obj))
plt.imshow(bessel, alpha = .1, interpolation = 'none', \
                                            extent = (\
                                            -1./step/2.*wavelength*f_obj,\
                                            1./step/2.*wavelength*f_obj,\
                                            -1./step/2.*wavelength*f_obj,\
                                            1./step/2.*wavelength*f_obj))
plt.xlabel(r'$\mu m$'), plt.ylabel(r'$\mu m$')
recap.add_subplot(234)
delta = 80
plt.imshow(annulus[annulus.shape[0]/2 - delta : annulus.shape[0]/2 + delta,\
                    annulus.shape[1]/2 - delta : annulus.shape[1]/2 + delta] * \
            stripes[stripes.shape[0]/2 - delta : stripes.shape[0]/2 + delta,\
                    stripes.shape[1]/2 - delta : stripes.shape[1]/2 + delta],\
            aspect = 'auto')
recap.add_subplot(235)
plt.imshow(annulus[annulus.shape[0]/2 - delta : annulus.shape[0]/2 + delta,\
                    annulus.shape[1]/2 - delta : annulus.shape[1]/2 + delta] * \
            stripes2[stripes.shape[0]/2 - delta : stripes.shape[0]/2 + delta,\
                    stripes.shape[1]/2 - delta : stripes.shape[1]/2 + delta],\
            aspect = 'auto')
recap.add_subplot(236)
plt.imshow(annulus[annulus.shape[0]/2 - delta : annulus.shape[0]/2 + delta,\
                    annulus.shape[1]/2 - delta : annulus.shape[1]/2 + delta] * \
            stripes3[stripes.shape[0]/2 - delta : stripes.shape[0]/2 + delta,\
                    stripes.shape[1]/2 - delta : stripes.shape[1]/2 + delta],\
            aspect = 'auto')
plt.show()
#-------------------------------------------------------------------------------
#sums
#-------------------------------------------------------------------------------
tot = np.mean(profile, axis = 1)
tot2 = np.mean(profile2, axis = 1)
tot3 = np.mean(profile3, axis = 1)

oneD = plt.figure('average')
oneD.add_subplot(311)
plt.plot(kx, tot, '.-', color = 'dodgerblue', label = '1')
plt.legend()
plt.grid()
oneD.add_subplot(312)
plt.plot(kx, tot2, '.-',color = 'orange', label = '2')
plt.grid()
plt.legend()
oneD.add_subplot(313)
plt.plot(kx, tot3, '.-', color = 'black', label = '3')
plt.grid()
plt.legend()
plt.xlabel(r'$\mu m$')
plt.show()

#-------------------------------------------------------------------------------
# Propagation simulation
#-------------------------------------------------------------------------------
 
# defocusDistance = np.linspace(-100, 100.1, 300)
# defocusPSF1 = np.zeros((annulus.shape[0], annulus.shape[1],defocusDistance.size))
# 
# pupil = annulus * stripes
# 
# for ctr, z in enumerate(defocusDistance):
#     # Add 0j to ensure that np.sqrt knows that its argument is complex
#     defocusPhaseAngle   = np.exp(2*np.pi*1j/wavelength*refr_index*z)*\
#                 np.exp(-1j*np.pi*wavelength*z*\
#                 ((X  / wavelength / f_obj)**2 + \
#                  (Y / wavelength / f_obj )**2))
#     defocusKernel       = defocusPhaseAngle
#     defocusPupil        = pupil * defocusKernel
#     defocusPSFA         = mtm.FT2(defocusPupil)
#     defocusPSF1[:,:,ctr] = np.real(defocusPSFA * np.conj(defocusPSFA))

























