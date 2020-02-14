#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 17:40:35 2019

@author: rafael
"""

"""
In the following example we show how to fit the diffusion kurtosis model on
diffusion-weighted multi-shell datasets and how to estimate diffusion kurtosis
based statistics.

First, we import all relevant modules:
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import dipy.reconst.dki as dki
import dipy.reconst.msdki as msdki
import dipy.reconst.dti as dti
import dipy.reconst.dki_micro as dki_micro
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from dipy.denoise.localpca import mppca
from dipy.denoise.gibbs import gibbs_removal

"""
DKI requires multi-shell data, i.e. data acquired from more than one non-zero
b-value. Here, we use fetch to download a multi-shell dataset which was kindly
provided by Hansen and Jespersen (more details about the data are provided in
their paper Hansen and Jespersen (2016)). The total size of the downloaded data
is 192 MBytes, however you only need to fetch it once.
"""

fraw, fbval, fbvec, t1_fname = get_fnames('cfin_multib')

data, affine = load_nifti(fraw)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

"""
Function ``get_fnames`` downloads and outputs the paths of the data,
``load_nifti`` returns the data as a nibabel Nifti1Image object, and
``read_bvals_bvecs`` loads the arrays containing the information about the
b-values and b-vectors. These later arrays are converted to the GradientTable
object required for Dipy's data reconstruction.

Before fitting the data, we preform some data pre-processing. We first compute
a brain mask to avoid unnecessary calculations on the background of the image.
"""

maskdata, mask = median_otsu(data, vol_idx=[0, 1], median_radius=4, numpass=2,
                             autocrop=False, dilate=1)

"""
Since the diffusion kurtosis models involves the estimation of a large number
of parameters [TaxCMW2015]_ and since the non-Gaussian components of the
diffusion signal are more sensitive to artefacts [NetoHe2012]_, it might be
favorable to suppress the effects of noise and artefacts before diffusion
kurtosis fitting. In this example the effects of noise are suppressed using the
Marcenko-Pastur PCA denoising algorithm (:ref:`example-denoise-mppca`) and
the Gibbs artefact suppression algorithm (:ref:`example-denoise-gibbs`). Due
to the dataset's large number of diffusion-weighted volumes, these algorithm
might take some hours to process.
"""

den, sig = mppca(data, patch_radius=4, return_sigma=True)
deng = gibbs_removal(den, slice_axis=2)

# img = nib.load('denoised_mppca.nii.gz')
# img = nib.load('data_mppca_gibbs.nii.gz')
# deng = img.get_fdata()


""" Before carrying on with dki processing, I will just visually inspect the
quality of data and preprocessing step. In the figure below, I am ploting a
slice of the raw and denoised + gibbs suppressed data (upper panels).  Lower
left panel show the difference between raw and denoised data - this difference
map does not present anatomicaly information, indicating that PCA denoising
suppression noise with minimal low of anatomical information. Lower right map
shows the noise std estimate. Lower noise std values are underestimated in
background since noise in background appraoches a Rayleigh distribution.  
"""

axial_slice = 10
vol = 100

fig1, ax = plt.subplots(2, 2, figsize=(12, 12),
                        subplot_kw={'xticks': [], 'yticks': []})

fig1.subplots_adjust(hspace=0.3, wspace=0.05)

ax.flat[0].imshow(data[:, :, axial_slice, vol].T, cmap='gray',
                  vmin=0, vmax=500, origin='lower')
ax.flat[0].set_title('raw')
ax.flat[1].imshow(deng[:, :, axial_slice, vol].T, cmap='gray',
                  vmin=0, vmax=500, origin='lower')
ax.flat[1].set_title('processed')
diff = den[:, :, axial_slice, vol].T - data[:, :, axial_slice, vol].T
ax.flat[2].imshow(diff, cmap='gray',
                  vmin=-10, vmax=10, origin='lower')
ax.flat[2].set_title('diff')
ax.flat[3].imshow(sig[:, :, axial_slice].T, cmap='gray',
                  vmin=0, vmax=10, origin='lower')
ax.flat[3].set_title('noise std')

"""
Now that we have loaded and pre-processed the data we can go forward
with DKI fitting. For this, the DKI model is first defined for the data's
GradientTable object by instantiating the DiffusionKurtosisModel object in the
following way:
"""

dkimodel = dki.DiffusionKurtosisModel(gtab)

# DTI model for comparison
dtimodel = dti.TensorModel(gtab)

# For MSDKI MSK estimates
msdkimodel = msdki.MeanDiffusionKurtosisModel(gtab)

# For WMTI
dki_micro_model = dki_micro.KurtosisMicrostructureModel(gtab)

"""
To fit the data using the defined model object, we call the ``fit`` function of
this object:
"""

# fitting raw data
dkifitraw = dkimodel.fit(data, mask)

# fitting processed data
dkifit = dkimodel.fit(deng, mask)

# fitting dti for reference
dtifit = dtimodel.fit(deng, mask)

# fitting msdki
msdkifit = msdkimodel.fit(deng, mask)

# fitting msdki
msdkifitraw = msdkimodel.fit(data, mask)


"""
The fit method creates a DiffusionKurtosisFit object, which contains all the
diffusion and kurtosis fitting parameters and other DKI attributes. For
instance, since the diffusion kurtosis model estimates the diffusion tensor,
all diffusion standard tensor statistics can be computed from the
DiffusionKurtosisFit instance. For example, we show below how to extract the
fractional anisotropy (FA), the mean diffusivity (MD), the axial diffusivity
(AD) and the radial diffusivity (RD) from the DiffusionKurtosisiFit instance.
"""

FAraw = dkifitraw.fa
MDraw = dkifitraw.md
ADraw = dkifitraw.ad
RDraw = dkifitraw.rd

FAden = dkifit.fa
MDden = dkifit.md
ADden = dkifit.ad
RDden = dkifit.rd

# dti parametric maps for reference
FAdti = dtifit.fa
MDdti = dtifit.md
ADdti = dtifit.ad
RDdti = dtifit.rd

"""
The DT based measures can be easily visualized using matplotlib. For example,
the FA, MD, AD, and RD obtained from the diffusion kurtosis model (upper
panels) and the tensor model (lower panels) are plotted for a selected axial
slice.
"""

axial_slice = 10

sc = 1000  # diffusion scaling

fig2, ax = plt.subplots(2, 4, figsize=(14, 7),
                        subplot_kw={'xticks': [], 'yticks': []})

fig2.subplots_adjust(hspace=0.3, wspace=0.1)

# Pre-processed DKI metrics
imdi = ax.flat[0].imshow(sc * MDden[:, :, axial_slice].T, cmap='gray',
                         vmin=0, vmax=2, origin='lower')
ax.flat[0].set_title('A.1) MD ($\mu m^2/ms$)')
ax.flat[1].imshow(sc * ADden[:, :, axial_slice].T, cmap='gray',
                  vmin=0, vmax=2, origin='lower')
ax.flat[1].set_title('A.2) AD ($\mu m^2/ms$)')
ax.flat[2].imshow(sc * RDden[:, :, axial_slice].T, cmap='gray',
                  vmin=0, vmax=2, origin='lower')
ax.flat[2].set_title('A.3) RD ($\mu m^2/ms$)')
imfa = ax.flat[3].imshow(FAden[:, :, axial_slice].T, cmap='gray',
                         vmin=0, vmax=0.7, origin='lower')
ax.flat[3].set_title('A.4) FA')

# Pre-processed DTI metrics
ax.flat[4].imshow(sc * MDdti[:, :, axial_slice].T, cmap='gray',
                  vmin=0, vmax=2, origin='lower')
ax.flat[4].set_title('B.1) MD ($\mu m^2/ms$)')
ax.flat[5].imshow(sc * ADdti[:, :, axial_slice].T, cmap='gray',
                  vmin=0, vmax=2, origin='lower')
ax.flat[5].set_title('B.2) AD ($\mu m^2/ms$)')
ax.flat[6].imshow(sc * RDdti[:, :, axial_slice].T, cmap='gray',
                   vmin=0, vmax=2, origin='lower')
ax.flat[6].set_title('B.3) RD ($\mu m^2/ms$)')
ax.flat[7].imshow(FAdti[:, :, axial_slice].T, cmap='gray',
                   vmin=0, vmax=0.7, origin='lower')
ax.flat[7].set_title('B.4) FA')

# add colorbars
a = 0
for lin in range(0, 2):
    for i in range(0, 4):
        if i == 3:
            fig2.colorbar(imfa, ax=ax.flat[a])
        else:
            fig2.colorbar(imdi, ax=ax.flat[a])
        a += 1

plt.show()
fig2.savefig('Diffusion_Tensor_Metrics.png', bbox_inches='tight')

"""
.. figure:: Diffusion_tensor_measures_from_DTI_and_DKI.png
   :align: center

   Diffusion tensor measures obtained from the diffusion tensor estimated
   from DKI (upper panels) and DTI (lower panels).

In addition to the standard diffusion statistics, the DiffusionKurtosisFit
instance can be used to estimate the non-Gaussian measures of mean kurtosis
(MK), the axial kurtosis (AK) and the radial kurtosis (RK).

Kurtosis measures are susceptible to high amplitude outliers. The impact of
high amplitude kurtosis outliers can be removed by introducing as an optional
input the extremes of the typical values of kurtosis. Here these are assumed to
be on the range between 0 and 3):
"""

# Raw data reference for standard metrics
MKraw = dkifitraw.mk(0, 3)
AKraw = dkifitraw.ak(0, 3)
RKraw = dkifitraw.rk(0, 3)

# DKI standard metrics
MKden = dkifit.mk(0, 3)
AKden = dkifit.ak(0, 3)
RKden = dkifit.rk(0, 3)

# DKI KFA and MKT
MKTden = dkifit.mkt(0, 3)
KFAden = dkifit.kfa
MKTraw = dkifitraw.mkt(0, 3)
KFAraw = dkifitraw.kfa

# MSDKI MSK
MSKden = msdkifit.msk
MSKraw = msdkifitraw.msk

"""
Now we are ready to plot the kurtosis standard measures using matplotlib:
"""
axial_slice = 10

fig3, ax = plt.subplots(2, 3, figsize=(12, 9),
                        subplot_kw={'xticks': [], 'yticks': []})

fig3.subplots_adjust(hspace=0.3, wspace=0.03)

imkurt = ax.flat[0].imshow(MKraw[:, :, axial_slice].T, cmap='gray',
                           vmin=0, vmax=1.5, origin='lower')
ax.flat[0].set_title('A.1) MK')
ax.flat[0].annotate('', fontsize=20, xy=(57, 30),
                    color='red',
                    xycoords='data', xytext=(30, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))
ax.flat[0].annotate('', fontsize=20, xy=(51, 62),
                    color='red',
                    xycoords='data', xytext=(30, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))


ax.flat[1].imshow(AKraw[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=1.5,
                  origin='lower')
ax.flat[1].set_title('A.2) AK')

ax.flat[2].imshow(RKraw[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=1.5,
                  origin='lower')
ax.flat[2].set_title('A.3) RK')
ax.flat[2].annotate('', fontsize=20, xy=(57, 30),
                    color='red',
                    xycoords='data', xytext=(30, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))
ax.flat[2].annotate('', fontsize=20, xy=(51, 62),
                    color='red',
                    xycoords='data', xytext=(30, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))

ax.flat[3].imshow(MKTraw[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=1.5,
                  origin='lower')
ax.flat[3].set_title('A.4) MKT')
ax.flat[3].annotate('', fontsize=20, xy=(58, 30),
                    color='red',
                    xycoords='data', xytext=(30, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))
ax.flat[3].annotate('', fontsize=20, xy=(53, 62),
                    color='red',
                    xycoords='data', xytext=(30, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))
imkfa = ax.flat[4].imshow(KFAraw[:, :, axial_slice].T, cmap='gray',
                          vmin=0, vmax=1, origin='lower')
ax.flat[4].set_title('A.5) KFA')

ax.flat[5].imshow(MSKraw[:, :, axial_slice].T, cmap='gray', vmin=0,
                  vmax=1.5, origin='lower')
ax.flat[5].set_title('A.6) MSK')

# add colorbars
a = 0
for lin in range(0, 3):
    for i in range(0, 2):
        if a == 4:
            fig3.colorbar(imkfa, ax=ax.flat[a])
        else:
            fig3.colorbar(imkurt, ax=ax.flat[a])
        a += 1

plt.show()
fig3.savefig('Kurtosis_Metrics_raw.png', bbox_inches='tight')

# Denoised data

axial_slice = 10

fig3, ax = plt.subplots(2, 3, figsize=(12, 8),
                        subplot_kw={'xticks': [], 'yticks': []})

fig3.subplots_adjust(hspace=0.1, wspace=0.1)

imkurt = ax.flat[0].imshow(MKden[:, :, axial_slice].T, cmap='gray',
                           vmin=0, vmax=1.5, origin='lower')
ax.flat[0].set_title('A) MK')
ax.flat[0].annotate('', fontsize=20, xy=(57, 30),
                    color='red',
                    xycoords='data', xytext=(30, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))
ax.flat[0].annotate('', fontsize=20, xy=(51, 62),
                    color='red',
                    xycoords='data', xytext=(30, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))


ax.flat[1].imshow(AKden[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=1.5,
                  origin='lower')
ax.flat[1].set_title('B) AK')

ax.flat[2].imshow(RKden[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=1.5,
                  origin='lower')
ax.flat[2].set_title('C) RK')
ax.flat[2].annotate('', fontsize=20, xy=(57, 30),
                    color='red',
                    xycoords='data', xytext=(30, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))
ax.flat[2].annotate('', fontsize=20, xy=(51, 62),
                    color='red',
                    xycoords='data', xytext=(30, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))

ax.flat[3].imshow(MKTden[:, :, axial_slice].T, cmap='gray', vmin=0, vmax=1.5,
                  origin='lower')
ax.flat[3].set_title('D) MKT')
ax.flat[3].annotate('', fontsize=20, xy=(58, 30),
                    color='red',
                    xycoords='data', xytext=(30, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))
ax.flat[3].annotate('', fontsize=20, xy=(53, 62),
                    color='red',
                    xycoords='data', xytext=(30, 0),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",
                                    color='red'))

imkfa = ax.flat[4].imshow(KFAden[:, :, axial_slice].T, cmap='gray',
                          vmin=0, vmax=1, origin='lower')
ax.flat[4].set_title('E) KFA')

ax.flat[5].imshow(MSKden[:, :, axial_slice].T, cmap='gray', vmin=0,
                  vmax=1.5, origin='lower')
ax.flat[5].set_title('F) MSK')


# add colorbars
a = 0
for lin in range(0, 3):
    for i in range(0, 2):
        if a == 4:
            fig3.colorbar(imkfa, ax=ax.flat[a])
        else:
            fig3.colorbar(imkurt, ax=ax.flat[a])
        a += 1

plt.show()
fig3.savefig('Kurtosis_Metrics.png', bbox_inches='tight')



"""
.. figure:: Kurtosis_tensor_standard_measures.png
   :align: center

   DKI standard kurtosis measures.

The non-Gaussian behaviour of the diffusion signal is expected to be higher
when tissue water is confined in multiple compartments. Therefore, MK is
higher in white matter due to the presence of myelin sheath that highly
compartmentalize different tissue compartments. These non-Gaussian effects are
expected to be more pronounced in prependicularly to white matter fibers and
thus the RK map presents higher amplitudes than the AK map.

The mean of the kurtosis tensor (MKT) produces
similar maps than the standard mean kurtosis (MK). On the other hand,
the kurtosis fractional anisotropy (KFA) maps shows that the kurtosis tensor
have different degrees of anisotropy than the FA measures from the diffusion
tensor.

=====================================================================
Reconstruction of the diffusion signal with the WMTI model
=====================================================================

DKI can also be used to derive concrete biophysical parameters by applying
microstructural models to DT and KT estimated from DKI

Before fitting this microstructural model, it is useful to indicate the
regions in which this model provides meaningful information (i.e. voxels of
well-aligned fibers). Following Fieremans et al. (2011), a simple way
to select this region is to generate a well-aligned fiber mask based on the
values of diffusion sphericity, planarity and linearity. Here we will follow
these selection criteria for a better comparison of our figures with the
original article published by Fieremans et al. (2011). Nevertheless,
it is important to note that voxels with well-aligned fibers can be selected
based on other approaches such as using predefined regions of interest.
"""

# Initialize well aligned mask with ones
well_aligned_mask = np.ones(data.shape[:-1], dtype='bool')

# Diffusion coefficient of linearity (cl) has to be larger than 0.4, thus
# we exclude voxels with cl < 0.4.
cl = dkifit.linearity.copy()
well_aligned_mask[cl < 0.4] = False

# Diffusion coefficient of planarity (cp) has to be lower than 0.2, thus
# we exclude voxels with cp > 0.3.
cp = dkifit.planarity.copy()
well_aligned_mask[cp > 0.3] = False

# Diffusion coefficient of sphericity (cs) has to be lower than 0.35, thus
# we exclude voxels with cs > 0.4.
cs = dkifit.sphericity.copy()
well_aligned_mask[cs > 0.4] = False

# Removing nan associated with background voxels
well_aligned_mask[np.isnan(cl)] = False
well_aligned_mask[np.isnan(cp)] = False
well_aligned_mask[np.isnan(cs)] = False

"""
Analogous to DKI, the data fit can be done by calling the ``fit`` function of
the model's object as follows:
"""

dki_micro_fit = dki_micro_model.fit(deng, mask=well_aligned_mask)

"""
The KurtosisMicrostructureFit object created by this ``fit`` function can then
be used to extract model parameters such as the axonal water fraction and
diffusion hindered tortuosity:
"""

AWF = dki_micro_fit.awf
TORT = dki_micro_fit.tortuosity

"""
These parameters are plotted below on top of the mean signal kurtosis maps:
"""


AWF0 = AWF.copy()
TORT0 = TORT.copy()

AWF[AWF0 == 0] = np.nan
TORT[TORT0 == 0] = np.nan

axial_slice = 10

fig4, axs = plt.subplots(2, 2, figsize=(10, 10),
                         subplot_kw={'xticks': [], 'yticks': []})

fig3.subplots_adjust(hspace=1, wspace=5)

axs.flat[0].imshow(MSKden[:, :, axial_slice].T, cmap=plt.cm.gray,
                   interpolation='nearest', origin='lower', vmin=0, vmax=2)
im0 = axs.flat[0].imshow(AWF[:, :, axial_slice].T, cmap=plt.cm.Reds, alpha=1,
                         vmin=0.3, vmax=0.7, interpolation='nearest',
                         origin='lower')
axs.flat[0].annotate('', fontsize=20, xy=(32, 38),
                     xycoords='data', xytext=(30, 0),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle="<-",
                                     color=[1, 1, 1]))
axs.flat[0].set_title('A.1) ')
fig4.colorbar(im0, ax=axs.flat[0], orientation='horizontal')

axs.flat[2].imshow(MSKden[:, :, axial_slice].T, cmap=plt.cm.gray,
                   interpolation='nearest', origin='lower', vmin=0, vmax=2)
im1 = axs.flat[2].imshow(TORT[:, :, axial_slice].T, cmap=plt.cm.Blues, alpha=1,
                         vmin=2, vmax=6, interpolation='nearest',
                         origin='lower')
axs.flat[2].annotate('', fontsize=20, xy=(32, 38),
                     xycoords='data', xytext=(30, 0),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle="<-",
                                     color=[1, 1, 1]))
axs.flat[2].set_title('B.1) ')
fig4.colorbar(im1, ax=axs.flat[2], orientation='horizontal')

# Ploting histogram
x = AWF0[well_aligned_mask].flatten()
x = x[~np.isnan(x)]
x = x[x > 0]
N, bins, patches = plt.hist(x, bins=20, range=(0, 1))
N = N / N.sum()
axs.flat[1].clear()
axs.flat[1].bar(bins[:-1] + 0.025, N, width=0.05,
                edgecolor=[0, 0, 0], color=[1, 0, 0])
axs.flat[1].set_xlabel('AWF')
axs.flat[1].set_title('A.2) ')
axs.flat[1].set_xlim(0, 1)

x = TORT0[well_aligned_mask].flatten()
x = x[~np.isnan(x)]
x = x[x > 0]
N, bins, patches = plt.hist(x, bins=20, range=(0, 10))
N = N / N.sum()
axs.flat[3].clear()
axs.flat[3].bar(bins[:-1] + 0.25, N, width=0.5,
                edgecolor=[0, 0, 0], color=[0, 0, 1])
axs.flat[3].set_xlabel('TORT')
axs.flat[3].set_title('B.2) ')
axs.flat[3].set_xlim(2, 10)

fig4.savefig('WMTI_Metrics.png', bbox_inches='tight')
plt.show()

"""
.. figure:: Kurtosis_Microstructural_measures.png
   :align: center

   Axonal water fraction (left panel) and tortuosity (right panel) values
   of well-aligned fiber regions overlaid on a top of a mean kurtosis all-brain
   image.


References
----------

.. [Fierem2011] Fieremans E, Jensen JH, Helpern JA (2011). White matter
                characterization with diffusion kurtosis imaging. NeuroImage
                58: 177-188
.. [Hansen2016] Hansen, B, Jespersen, SN (2016). Data for evaluation of fast
                kurtosis strategies, b-value optimization and exploration of
                diffusion MRI contrast. Scientific Data 3: 160072
                doi:10.1038/sdata.2016.72

.. include:: ../links_names.inc
"""

