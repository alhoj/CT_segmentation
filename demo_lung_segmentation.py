import lung_segmentation
import numpy as np
from glob import glob
import nibabel as nib
import os
import matplotlib.pyplot as plt

plt.close('all')
path_in = './Images/'
path_out = './Lungs/'

## Find all images
paths = glob('%s/*.nii.gz' % path_in)
print('Images found:', len(paths))

## Loop over images, save lung masks and show lung areas in mm2
# first, check if output folder exists; if not, create
if not os.path.exists(path_out):
    os.mkdir(path_out)
for path in paths:
    img_name = path.split("\\")[-1].split('.nii')[0]
    ct_img = nib.load(path)
    ct_array = ct_img.get_fdata()
    # find countours in the image
    contours = lung_segmentation.intensity_seg(ct_array)
    # find countour corresponding to the lungs
    lungs = lung_segmentation.find_lungs(contours)
    # plot contour on top of the CT image
    lung_segmentation.show_contour(ct_array, lungs, save='%s/%s.png' % (path_out, img_name))
    # create mask from lung contours
    lung_mask = lung_segmentation.mask_from_contours(ct_array, lungs)
    # compute lung area in mm2
    pixdim = lung_segmentation.find_pix_dim(ct_img)
    lung_area = lung_segmentation.compute_area(lung_mask, pixdim)
    print('%s, lung area %.1f mm^2' % (img_name, lung_area))