"""
The :mod:`data_factory` module generates and saves data for training,
validation and testing"""

# Author: Hector Loarca, Ingo Guehring
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import h5py
import glob
import pickle
import os
import math
import numpy as np
import numpy.random as rnd
from skimage.transform import resize
from adler.odl.phantom import random_phantom as random_phantom_jonas

from ellipse.ellipseWF_factory import random_phantom, WFupdate
from realphantom.realphantomWF_factory import random_realphantom

from shared.shared import create_increasing_dir


# Generate random phantom randomizing the ellipses number and directional bias
def random_phantom_generation(size, nClasses):
    """Create a pair of phantom and WFimage image with phantom with random
    ellipses of size `(size,size)`, and nClasses in the wavefront set

    Parameters
    -----------
    size : integer, size of image

    nClasses : integer, the number of orientation in the wavefront set
        Returns
    -----------
    phantom : numpy array, `size` x `size` image with `nEllipses`
         phantom with random ellipses
         
    WFimage : numpy array, `size` x `size` image with the wavefront set
    """
    nEllipses = np.random.randint(10,20)
    dirBias = np.random.randint(0,180)
    phantom, _, _,  WFimage = random_phantom(size, nEllipses, dirBias, nClasses)
    return phantom, WFimage


# Generate random phantom randomizing the ellipses number and directional bias
def random_phantom_generation_lowd(size, nClasses, lowd):
    """Create a pair of phantom and WFimage image with phantom with random
    ellipses of size `(size,size)`, and nClasses in the wavefront set

    Parameters
    -----------
    size : integer, size of image

    nClasses : integer, the number of orientation in the wavefront set
    
    lowd : The number of equally spaced angles you are going to measure

        Returns
    -----------
    phantom : numpy array, `size` x `size` image with `nEllipses`
         phantom with random ellipses
    
    WFimage : numpy array, `size` x `size` image with the wavefront set
    
    WFimage_lowd : numpy array, `size` x `size` image with the low dose wavefront set
    """
    # Compute first the full dose Wavefront set
    nEllipses = np.random.randint(10,20)
    dirBias = np.random.randint(0,180)
    phantom, WFpoints, WFclasses,  WFimage = random_phantom(size, nEllipses, dirBias, nClasses)
    
    # Compute low dose WF image
    angles_lowd = np.array([i for i in range(0,180,int(180/lowd))])
    # Extracting the wavefront set orientations in the low dose
    angles_gt = (np.array(WFclasses).astype(int)[:,0]-1)
    angles_gt_lowd = np.array([angle in angles_lowd for angle in angles_gt])
    # Generating the new WFpoints and classes
    WFpoints_gt_lowd = WFpoints[angles_gt_lowd]
    WFclasses_gt_lowd = list(np.array(WFclasses)[angles_gt_lowd])
    # Generating the low dose WFimage
    WFimage_lowd = np.zeros([size,size])
    WFimage_lowd = WFupdate(WFpoints_gt_lowd, WFclasses_gt_lowd, WFimage_lowd)
    
    return phantom, WFimage, WFimage_lowd

def random_realphantom_generation(size, nClasses):
    nRegions = np.random.randint(5,10)
    npoints_max = np.random.randint(8,15)
    realphantom, _, _,  WFimage = random_realphantom(size, nRegions, npoints_max, nClasses)
    return realphantom, WFimage

def random_realphantom_generation_lowd(size, nClasses, lowd):
    # Compute first the full dose Wavefront set
    nRegions = np.random.randint(5,10)
    npoints_max = np.random.randint(8,15)
    realphantom, WFpoints, WFclasses,  WFimage = random_realphantom(size, nRegions, npoints_max, nClasses)
    
    # Compute low dose WF image
    angles_lowd = np.array([i for i in range(0,180,int(180/lowd))])
    # Extracting the wavefront set orientations in the low dose
    angles_gt = (np.array(WFclasses).astype(int)[:,0]-1)
    angles_gt_lowd = np.array([angle in angles_lowd for angle in angles_gt])
    # Generating the new WFpoints and classes
    WFpoints_gt_lowd = WFpoints[angles_gt_lowd]
    WFclasses_gt_lowd = list(np.array(WFclasses)[angles_gt_lowd])
    # Generating the low dose WFimage
    WFimage_lowd = np.zeros([size,size])
    WFimage_lowd = WFupdate(WFpoints_gt_lowd, WFclasses_gt_lowd, WFimage_lowd)
    
    return realphantom, WFimage, WFimage_lowd


# Batch generation for training
def generate_data_WF(batch_size, size, nClasses):
    """Create a batch of pairs of phantom and WFimage image with phantom 
    with random ellipses of size `(size,size)`, nClasses in the wavefront set
    and batch size `batch_size`

    Parameters
    -----------
    batch_size : integer, size of the batch
    
    size : integer, size of image

    nClasses : integer, the number of orientation in the wavefront set

        Returns
    -----------
    Batch of
    
    phantom : numpy array, `size` x `size` image with `nEllipses`
         phantom with random ellipses
         
    WFimage : numpy array, `size` x `size` image with the wavefront set
    """
    n_generate = batch_size

    y_arr = np.empty((n_generate, size, size, 1), dtype='float32')
    x_true_arr = np.empty((n_generate, size, size, 1), dtype='float32')
    
    for i in range(n_generate):
        phantom, WFimage = random_phantom_generation(size, nClasses)
        
        x_true_arr[i, ..., 0] =  WFimage
        y_arr[i, ..., 0] = phantom
    return y_arr, x_true_arr

# Batch generation for training
def generate_data_WFinpaint(batch_size, size, nClasses, lowd):
    """Create a batch of pairs of full dose WFimage and low dose WFimage from a phantom
    with random ellipses of size `(size,size)`, nClasses in the wavefront set
    and batch size `batch_size`

    Parameters
    -----------
    batch_size : integer, size of the batch
    
    size : integer, size of image

    nClasses : integer, the number of orientation in the wavefront set
    
    lowd : The number of equally spaced angles you are going to measure

        Returns
    -----------
    Batch of
    
    phantom : numpy array, `size` x `size` image with `nEllipses`
         phantom with random ellipses
         
    WFimage : numpy array, `size` x `size` image with the wavefront set
    """
    n_generate = batch_size

    y_arr = np.empty((n_generate, size, size, 1), dtype='float32')
    x_true_arr = np.empty((n_generate, size, size, 1), dtype='float32')
    
    for i in range(n_generate):
        _, WFimage, WFimage_lowd = random_phantom_generation_lowd(size, nClasses, lowd)
        
        x_true_arr[i, ..., 0] =  WFimage
        y_arr[i, ..., 0] = WFimage_lowd
    return y_arr, x_true_arr

# Batch generation for training
def generate_realphantom_WF(batch_size, size, nClasses):
    n_generate = batch_size

    y_arr = np.empty((n_generate, size, size, 1), dtype='float32')
    x_true_arr = np.empty((n_generate, size, size, 1), dtype='float32')
    
    for i in range(n_generate):
        phantom, WFimage = random_realphantom_generation(size, nClasses)
        
        x_true_arr[i, ..., 0] =  WFimage
        y_arr[i, ..., 0] = phantom
    return y_arr, x_true_arr

def generate_realphantom_WFinpaint(batch_size, size, nClasses, lowd):
    n_generate = batch_size

    y_arr = np.empty((n_generate, size, size, 1), dtype='float32')
    x_true_arr = np.empty((n_generate, size, size, 1), dtype='float32')
    
    for i in range(n_generate):
        _, WFimage, WFimage_lowd = random_realphantom_generation_lowd(size, nClasses, lowd)
        
        x_true_arr[i, ..., 0] =  WFimage
        y_arr[i, ..., 0] = WFimage_lowd
    return y_arr, x_true_arr

# Keras data generator 
def DataGenerator_WF(batch_size, size, nClasses):
    while True:
        yield generate_data_WF(batch_size, size, nClasses)
        
def DataGenerator_WFinpaint(batch_size, size, nClasses, lowd):
    while True:
        yield generate_data_WFinpaint(batch_size, size, nClasses, lowd)
        
def DataGenerator_realphantom_WF(batch_size, size, nClasses):
    while True:
        yield generate_realphantom_WF(batch_size, size, nClasses)
        
def DataGenerator_realphantom_WFinpaint(batch_size, size, nClasses, lowd):
    while True:
        yield generate_realphantom_WFinpaint(batch_size, size, nClasses, lowd)