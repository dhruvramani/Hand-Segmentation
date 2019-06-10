import cv2
import os, sys, glob, time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as matimg
from skimage.segmentation import mark_boundaries
import imsegm.utilities.data_io as tl_data
import imsegm.pipelines as segm_pipe

sys.path += [os.path.abspath('.'), os.path.abspath('..')] # Add path to root

def segment(filepath, outfilepath):
    img = np.array(Image.open(filepath))
    nb_classes = 3
    sp_size = 25
    sp_regul = 0.2
    dict_features = {'color': ['mean', 'std', 'median']}
    model, _ = segm_pipe.estim_model_classes_group([img], nb_classes, sp_size=sp_size, sp_regul=sp_regul, dict_features=dict_features, pca_coef=None, model_type='GMM')
    dict_debug = {}
    seg, _ = segm_pipe.segment_color2d_slic_features_model_graphcut(img, model, sp_size=sp_size, sp_regul=sp_regul, dict_features=dict_features, gc_regul=5., gc_edge_type='color', debug_visual=dict_debug)
    kernel = np.ones((5 , 5),np.float32)/25
    dst = cv2.filter2D(seg, -1, kernel)
    matimg.imsave(outfilepath, seg)

if __name__ == '__main__':
    segment("./test_images/test8.jpg", "./hand_erod.jpg")
