import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.feature import canny
from skimage.exposure import histogram
from skimage.morphology import watershed
from skimage.io import imread, imsave, imshow

def segment(filepath, outfilepath):
    image = imread(filepath, as_gray=True)
    hist, hist_centers = histogram(image)
    edges = canny(image/255.)
    fill_edges = ndi.binary_fill_holes(edges)
    
    label_objects, nb_labels = ndi.label(fill_edges)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 20
    mask_sizes[0] = 0
    hand_cleaned = mask_sizes[label_objects]

    markers = np.zeros_like(image)
    markers[image < 30] = 0
    markers[image > 100] = 1

    elevation_map = sobel(image)
    segmentation = watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    imsave(outfilepath, segmentation)

if __name__ == '__main__':
    segment("./test_images/hand2.jpg", "./hand2_erod.jpg")