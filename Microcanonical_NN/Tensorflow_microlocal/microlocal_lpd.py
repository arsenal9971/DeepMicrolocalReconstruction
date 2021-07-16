# Important libraries

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.ndimage as ndimage
from tensorflow.nn import relu

### Microlocal analysis of 

# Finite differences matrices
D_11 = np.array([[0,0,0],[0, 1, 0], [0 , 0 , 0]])
D_12 = np.array([[0,1,0],[0, 0, 0], [0 , -1 , 0]])
D_21 = np.array([[0,0,0],[1, 0, -1], [0 , 0 , 0]])
D_22 = np.array([[1,0,-1],[0, 0, 0], [-1 , 0 , 1]])
D_13 = np.array([[0,-1,0],[0, 2, 0], [0 , -1 , 0]])
D_31 = np.array([[0,0,0],[1, -2, 1], [0 , 0 , 0]])
D_23 = np.array([[1,-2,1],[0, 0, 0], [-1 , 2 , -1]])
D_32 = np.array([[1,0,-1],[-2, 0, 2], [1 , 0 , -1]])
D_33 = np.array([[-1,2,-1],[2, -4, 2], [-1 , 2 , -1]])

D = [D_11, D_12, D_13, D_21, D_22, D_23, D_31, D_32, D_33]

# Let us define the matrix for the change of coordinates
A = np.array([
    [0, 0, 0,  0, 1, 1, 0, 1, -1],
    [0, 1, -1, 0, 0, -2, 0, 0, 2],
    [0, 0, 0, 0, -1, 1, 0, -1, -1],
    [0, 0, 0, 1, 0, 0, 1, -2, 2],
    [1, 0, 2, 0, 0, 0, -2, 0, -4],
    [0, 0, 0, -1, 0, 0, 1, 2, 2],
    [0, 0, 0, 0, -1, -1, 0, 1, -1],
    [0, -1, -1, 0, 0, 2, 0, 0, 2],
    [0, 0, 0, 0, 1, -1, 0, -1, -1]
])

# Inverse
Ainv = np.linalg.inv(A)

def pT(B, Xi): 
    p = (B[0,0]+B[0,1]*Xi[1]+B[1,0]*Xi[0]+B[1,1]*Xi[0]*Xi[1]+
        B[0,2]*Xi[1]**2+B[2,0]*Xi[1]**2+B[1,2]*Xi[0]*Xi[1]**2+
        B[2,1]*Xi[1]*Xi[0]**2+B[2,2]*(Xi[1]**2)*Xi[0]**2)
    return p

def ellipt_layer_tf(name_layer, Ainv_tf, radon, gr):
    if "dual" in name_layer:
        shape = radon.range.shape
    else: 
        shape = radon.domain.shape
    kernels = tf.transpose(gr.get_tensor_by_name(name_layer),[2,0,1,3])
    kernels_shape = kernels.shape
    kernels = tf.reshape(kernels, [kernels_shape[0],kernels_shape[1]*kernels_shape[2], kernels_shape[3]])
    Bflats = tf.tensordot(Ainv_tf,kernels, axes = [0,1])
    Bs = tf.reshape(Bflats,[3,3,kernels_shape[0], kernels_shape[3]]).eval()
    ellipts = []
    ellipts = []
    for n_value in range(Bs.shape[2]):
        for channel in range(Bs.shape[3]):
            B = Bs[:,:, n_value, channel]
            ellipts.append(np.min(np.abs(np.array([pT(B,np.array([xi,yi])) 
                                             for xi in range(shape[0]) for yi in 
                                             range(shape[1])]))));    
        return np.mean(ellipts)
    

def ellipt_layer_numpy(name_layer, Ainv, radon, gr):
    if "dual" in name_layer:
        shape = radon.range.shape
    else: 
        shape = radon.domain.shape
    kernels = gr.get_tensor_by_name(name_layer).eval()
    ellipts = []
    for n_value in range(kernels.shape[2]):
        for channel in range(kernels.shape[3]):
            kernel= kernels[:,:,n_value,channel]
            Bflat = Ainv.dot(kernel.flatten())
            B = Bflat.reshape(3,3);
            ellipts.append(np.min(np.abs(np.array([pT(B,np.array([xi,yi])) 
                                             for xi in range(shape[0]) for yi in 
                                             range(shape[1])]))));
    return np.mean(ellipts)

### Microlocal analysis of the ReLU function

# Function that computes the gradient for each channel
def grad_channel_batch(f):
    fxs = np.zeros(f.shape)
    fys = fxs.copy(f.shape)
    for batch in range(fxs.shape[0]):
        for channel in range(fxs.shape[3]):
            fi = f[batch,:,:,channel]
            fxs[batch,:,:,channel] = ndimage.sobel(fi, axis= 0, mode='constant')
            fys[batch,:,:,channel] = ndimage.sobel(fi, axis= 1, mode='constant')
    return fxs, fys

# Compute WF set from gradient
def WF_grad(fx,fy):
    WFset = np.zeros(fx.shape)
    for i in range(fx.shape[0]):
        for j in range(fx.shape[1]):
            if fx[i,j] == 0:
                WFset[i,j] = 0
            else:
                WFset[i,j] = 180*np.arctan(fy[i,j]/fx[i,j])/(2*np.pi)
    return WFset

# Compute WF set from gradient for mutiple channels
def WF_grad_channel_batch(fxs, fys):
    WFset = np.zeros(fxs.shape)
    for batch in range(fxs.shape[0]):
        for channel in range(fxs.shape[3]):
            fx = fxs[batch,:,:,channel]
            fy = fys[batch,:,:,channel]
            WFset[batch,:,:,channel] = WF_grad(fx,fy)
    return WFset

# Relevan ReLU definition

def ReLU(x):
    return np.array([max(xi,0) for xi in x])

def ReLU2(f):
    ReLUf=np.zeros(f.shape);
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            ReLUf[i,j] = max(f[i,j],0)
    return ReLUf

def ReLU2_channel_batch(f):
    ReLUfs = np.zeros(f.shape)
    for batch in range(f.shape[0]):
        for channel in range(f.shape[3]):
            fi = f[batch,:,:,channel]
            ReLUfs[batch,:,:,channel] = ReLU2(fi);
    return ReLUfs

def ReLU_microlocal(update, radon, WFset_f, gr, iterate, conv_layer, dual_layer):
    ## Defining shape and name of the layer
    if dual_layer== True:
        ld = 'dual'
        shape = radon.range.shape
    else:
        ld = 'primal'
        shape = radon.domain.shape

    if conv_layer == 0:
        conv_name = '/conv2d/'
    else:
        conv_name = '/conv2d_'+str(conv_layer)+'/'

    # Defining name of the kernel
    name_kernel = (ld+'_iterate_'+str(iterate)+conv_name+'kernel:0')

    # Load the kernels value
    kernels = gr.get_tensor_by_name(name_kernel)
    
    # Apply the convolutional layer to the update
    conv_kernel = tf.nn.conv2d(update, kernels, [1, 1, 1, 1], padding='SAME')
    conv_relu = relu(conv_kernel)
    # Evaluate convolutional kernel
    f = conv_kernel.eval()

    # Apply heaviside function
    Hf = np.heaviside(f,0)
    
    # Apply ReLU output for each batch and channel
    ReLUfs = ReLU2_channel_batch(f)

    # Computing the Heavisde part of the Wavefrontset 
    fxs, fys = grad_channel_batch(Hf)
    WFset_Hf = WF_grad_channel_batch(fxs, fys)
    
    # Computing the Wavefrnt set of the ReLU layer
    WFset_ReLUf = (WFset_Hf+WFset_f)%180
    
    return name_kernel, conv_relu, WFset_ReLUf