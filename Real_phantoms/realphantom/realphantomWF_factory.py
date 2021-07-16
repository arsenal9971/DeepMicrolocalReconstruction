"""
The :mod:`realphantom` module implements a class which handles the creation of the
random realisticphantom"""
# Author: Hector Loarca (help of Ingo Guehring from another joint project,
#                        and help of Jonas Adler for point sampling)

# next two lines might work/be necessary only for mac
import matplotlib
matplotlib.use('Agg')
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

from ellipse.ellipseWF_factory import _fig2data, WFupdate,_opacity_gen, _center_origin_gen
import matplotlib.patches as patches
import scipy.interpolate as interpolate
from scipy.ndimage import gaussian_filter

def smooth_edge(x,y, k):
    # Computing the spline
    S = interpolate.UnivariateSpline(x, y, k=k)
    xnew = np.linspace(x[0], x[-1], 
                       num=2*int((x[-1]-x[0])), endpoint=True)
    ynew = S(xnew)
    WFpoints = np.zeros((xnew.shape[0],2));
    WFpoints[:,0] = xnew;
    WFpoints[:,1] = ynew;
    # Computing the classes
    WFclasses = [np.array([(90-np.arctan(S.derivative(n=1)(xi))
              *180/(np.pi))%180+1]) for xi in xnew];
    return WFpoints, WFclasses

def _region_gen(size, verts, opacity, grad_level,nClasses):
    ## Wavefront set generation
    npoints = verts.shape[0]
    WFpoints_all = []
    WFclasses_all = []
    i = 0
    while i<npoints-1:
        XY = [verts[i,:]]
        # Selection of degree
        k = np.random.randint(1,4)
        # Reducing the degree if there are no enough points to fit 
        while npoints-i-1 <= k-1:
            k = k-1
        j = 1
        indices = np.array([0])
        # Check repeated x-entries
        xrep = (len(np.array(XY)[:,0]))== len(set(np.array(XY)[:,0]))  
        while (j < k+1) and (((indices[0] == 0)or(indices[-1]==0))) and (xrep):
            XY += [verts[i+j,:]]
            # Reordering the verticese
            if (i+j+1)< npoints:
                xnext = np.array(XY + [verts[i+j+1,:]])[:,0]
                indices = np.argsort(xnext)
                xrep = (len(list(xnext))== len(set(xnext)))
            j+=1
        XY = np.array(XY)
        indices = np.argsort(XY[:,0])
        XY = XY[indices,:]
        k = XY.shape[0]-1
        # Getting the spline
        x = XY[:,0]
        y = XY[:,1]
        WFpoints, WFclasses = smooth_edge(x,y, k)
        # Correcting order 
        if indices[-1] == 0:
            WFpoints = np.flip(WFpoints, axis=0)
            WFclasses.reverse()
        WFpoints_all+=list(WFpoints)
        WFclasses_all += WFclasses
        i += k
        
    WFpoints_all = np.array(WFpoints_all)
    keep_index = (WFpoints_all[:,0] < size)*(WFpoints_all[:,-1] < size) *(WFpoints_all[:,0] > 0)*(WFpoints_all[:,-1] > 0)
    WFclasses_all = list(np.array(WFclasses_all)[keep_index])
    WFpoints_all = WFpoints_all[keep_index]
    
    ## Correcting classes
    WFclasses_all = [np.array([(int(classe*nClasses/180)+1)])
               for classe in WFclasses_all]
    
    WFimage = np.zeros((size,size))
    WFimage = WFupdate(WFpoints_all, WFclasses_all, WFimage)
    WFimage = WFimage
    
    ## Generate the region
    
    patch = plt.Polygon([[WFpoints_all[i,0], WFpoints_all[i,1]] for i in range(len(WFpoints_all))])
    fig = plt.figure(0, frameon=False, figsize=(1, 1), dpi=size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.add_patch(patch)
    patch.set_clip_box(ax.bbox)
    patch.set_alpha(None)
    patch.set_facecolor(np.zeros(3))

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    fig.add_axes(ax)
    plt.axis('off')

    plt.close(fig)
    # Convert figure to data
    region = _fig2data(fig)
    plt.close(fig)
    # Take just the first color entry
    region = np.flip(region[:, :, 1], axis=0)
    # Normalize the data
    region = region/region.max();
    region = 1-(((region-1)*opacity)+1)
    if grad_level >=0:
        region = gaussian_filter(region, sigma = 5*grad_level)
    return region, np.array(WFpoints_all), WFclasses_all, WFimage

def _region_verts_gen(size, npoints_max = 15):
    # magnitude of the perturbation from the unit circle,
    r =  rnd.random()
    scale = rnd.randint(low=10, high= int(size/4))
    # Random parameters for the ellipse
    center = _center_origin_gen(size)
    # Number of points
    npoints = rnd.randint(low=5, high= npoints_max)
    # Generate the points
    angles = np.linspace(np.pi,-np.pi,npoints)
    verts = np.stack((np.cos(angles),np.sin(angles))).T*(2*r*np.random.random(npoints)+1-r)[:,None]
    verts[-1,:] = verts[0,:]
    # rescaling and translating vertices
    verts = scale*verts;
    verts = verts+center;
    return verts

def random_phantom(size, nRegions, npoints_max, nClasses):
    """Create a `size` x `size` image of a realphantom with  `nRegions` random
    regions

    Parameters
    -----------
    size : integer, size of image

    nRegions : integer, the number of regions in the image

    npoints_max : integer, the number of maximum points to generate regions
    
    nClasses : integer, the number of classes

        Returns
    -----------
    realphantom : numpy array, `size` x `size` image with `nRegions`
         phantom with random regions

    WFpoints_all : numpy array, with the wavefront set point locaion
    
    WFclasses_all : list of numpya arrays, with the classes for each wavefront set point
    
    WFimage : numpy array, with the wavefront set image
    """
    # Create the WFimage, WFpoints and WF_classes
    WFimage = np.zeros((size,size))
    WFpoints_all = []
    WFclasses_all = []
    realphantom = np.zeros((size,size))
    
    # Generate the regions
    for i in range(nRegions):
        verts = _region_verts_gen(size, npoints_max);
        opacity = _opacity_gen()
        grad_level = rnd.uniform(-2,2)
        region, WFpoints, WFclasses, _ = _region_gen(size, verts, opacity, grad_level,nClasses)
        
        keep_index = (WFpoints[:,0] < size)*(WFpoints[:,1] < size) 
        WFclasses = list(np.array(WFclasses)[keep_index])
        WFpoints = WFpoints[keep_index] 
    

        WFpoints_all += list(WFpoints)
        WFclasses_all += list(WFclasses)
        WFimage = WFupdate(WFpoints, WFclasses, WFimage)
        
        realphantom += region;
        
    # Get the normalization of the total realphantom
    realphantom = realphantom/realphantom.max()
    realphantom = np.interp(realphantom, (realphantom.min(), realphantom.max()), (0, 1))
    
    return realphantom, np.array(WFpoints_all), WFclasses_all, WFimage