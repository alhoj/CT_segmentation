import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from skimage import measure
from glob import glob
import csv

def plot_slice(ct_numpy, level=None, window=None):
    """
    Function to display a CT image slice

    Parameters
    ----------
    ct_numpy : ndarray
        Image as numpy 2D array.
    level : int | None
        Central intensity in Hounsfield units. If None (default), show the image with the whole unclipped range.
    window : int | None
        Window width in Hounsfield units. If None (default), show the image with the whole unclipped range.
    """
    if level and window:
        max = level + window/2
        min = level - window/2
        ct_numpy = ct_numpy.clip(min,max)
    # fig = plt.figure()
    fig = plt.imshow(ct_numpy.T, cmap="gray", origin="lower")
    return fig
    

def intensity_seg(ct_numpy, min=-1000, max=-300):
    """
    Find constant valued contours using marching squares method.
    Parameters
    ----------
    ct_numpy : ndarray
        Image as numpy 2D array.
    min : int
        Minimum intensity in Hounsfield units (default -1000).
    max : int
        Maximum intensity in Hounsfield units (default -300).

    Returns
    ------- 
    contours : list of arrays
        List of found contours.
    """
    clipped = ct_numpy.clip(min, max)
    clipped[clipped != max] = 1
    clipped[clipped == max] = 0
    contours = measure.find_contours(clipped, 0.95)
    return contours


def contour_distance(contour):
    """
    Given a set of points that may describe a contour
     it calculates the distance between the first and the last point
     to infer if the set is closed.
    Parameters
    ----------
    contour: list of arrays
        All found contours in the image.

    Returns
    -------
    distance : float
        Euclidean distance between first and last point
    """
    dx = contour[0, 1] - contour[-1, 1]
    dy = contour[0, 0] - contour[-1, 0]
    distance = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
    return distance


def find_lungs(contours):
    """
    Chooses the contours that correspond to the lungs and the body.
    First, exclude non-closed contours. Then, assume minimum area and volume to exclude small contours.
    Then, exclude body as the highest volume closed countour. The remaining area correspond to the lungs.

    Parameters
    ----------
    contours : list of arrays
        All found contours in the image.

    Returns
    -------
    lung_contour : ndarray
        Contours that correspond to the lung area.

    """
    body_and_lung_contours = []
    vol_contours = []

    for contour in contours:
        hull = ConvexHull(contour)

        # minimum volume of 2000 pixels assumed
        if hull.volume > 2000 and contour_distance(contour) < 1:
            body_and_lung_contours.append(contour)
            vol_contours.append(hull.volume)

    if len(body_and_lung_contours) == 2:
        lung_contours = body_and_lung_contours
    elif len(body_and_lung_contours) > 2:
        vol_contours, body_and_lung_contours = (list(t) for t in
                                                zip(*sorted(zip(vol_contours, body_and_lung_contours))))
        body_and_lung_contours.pop(-1)
        lung_contours = body_and_lung_contours
    return lung_contours
    

def show_contour(ct_numpy, contours, save=False):
    """
    Plots contours on top of CT image.

    Parameters
    ----------
    ct_numpy : ndarray
        Image as numpy 2D array.
    contours : ndarray | list of arrays
        Contours
    save : bool | str
        Save plot (default False). If str, saves the plot using the str as path.

    Returns
    -------
    lung_contour : ndarray
        Contour that correspond to the lung area.

    """
    # fig, ax = plt.subplots()
    plot_slice(ct_numpy, -600, 1500)
    ax = plt.gca()
    # ax.imshow(ct_numpy.T, cmap=plt.cm.gray)
    if type(contours) != list: contours = [contours]
    for contour in contours:
        ax.plot(contour[:, 0], contour[:, 1], linewidth=2, color='r')

    ax.set_xticks([])
    ax.set_yticks([])

    if save:
        plt.savefig(save)
        plt.close()


def mask_from_contours(ct_numpy, contours):
    """
    Creates a binary mask with the dimensions of the image from the contours.

    Parameters
    ----------
    ct_numpy : ndarray
        Image as numpy 2D array.
    contours : list of arrays
        Contours

    Returns:
    lung_mask : ndarray
        Binary mask
    """

    lung_mask = np.array(Image.new('L', ct_numpy.shape, 0))
    for contour in contours:
        x = contour[:, 0]
        y = contour[:, 1]
        polygon_tuple = list(zip(x, y))
        img = Image.new('L', ct_numpy.shape, 0)
        ImageDraw.Draw(img).polygon(polygon_tuple, outline=0, fill=1)
        mask = np.array(img)
        lung_mask += mask
    return lung_mask.T  # transpose it to be aligned with the image dims



def find_pix_dim(ct_img):
    """
    Get the pixdim of CT image.
    A general solution that get the pixdim indicated from the image dimensions. 
    From the last two image dimensions we get their pixel dimension.
    Parameters
    ----------
    ct_img : nib image

    Returns
    ------- 
    pixDims : list
        List of X and Y pixel dimensions.
    """
    pix_dim = ct_img.header["pixdim"]
    dim = ct_img.header["dim"]
    max_indx = np.argmax(dim)
    pixdimX = pix_dim[max_indx]
    dim = np.delete(dim, max_indx)
    pix_dim = np.delete(pix_dim, max_indx)
    max_indy = np.argmax(dim)
    pixdimY = pix_dim[max_indy]
    pixDims = [pixdimX, pixdimY]
    return pixDims


def compute_area(mask, pixdim):
    """
    Computes the area (number of pixels) of a binary mask and multiplies the pixels
    with the pixel dimension of the CT image.
    Parameters
    ----------
    lung_mask : binary lung mask
    pixdim : list or tuple with two values

    Returns
    ------- 
    lung_area : float
        The lung area in mm^2
    """
    mask[mask >= 1] = 1
    lung_pixels = np.sum(mask)
    lung_area = lung_pixels * pixdim[0] * pixdim[1]
    return lung_area