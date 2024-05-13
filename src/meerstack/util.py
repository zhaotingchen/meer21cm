import numpy as np
from astropy import constants,units
from astropy.coordinates import SkyCoord

def get_wcs_coor(wcs,xx,yy):
    coor = wcs.pixel_to_world(xx,yy)
    ra = coor.ra.deg.T
    dec = coor.dec.deg.T
    return ra,dec

def PCAclean(M,N_fg,w=None,W=None,returnAnalysis=False,MeanCentre=False,los_axis=-1,return_A=False):
    '''
    Performs PCA cleaning of the map data.
    '''
    if los_axis < 0:
        # change -1 to 2
        los_axis = 3+los_axis
    # make sure los is the fist axis
    axes = [0,1,2]
    axes.remove(los_axis)
    axes = [los_axis,]+axes
    # transpose map data
    M = np.transpose(M,axes=axes)
    nz,nx,ny = M.shape
    M = M.reshape((len(M),-1))
    if W is not None:
        W = np.transpose(W,axes=axes)
        W = W.reshape((len(M),-1))
    if w is not None:
        w = np.transpose(w,axes=axes)
        w = W.reshape((len(M),-1))
    # this is weird. Why are there two weights?
    if MeanCentre:
        if W is None:
            M = M - np.mean(M,1) # Mean centre data
        else:
            M = M - np.sum(M*W,1)[:,None]/np.sum(W,1)[:,None]
    ### Covariance calculation:
    if w is None:
        w = 1.0
    C = np.cov(w*M) # include weight in frequency covariance estimate
    if returnAnalysis==True:
        eigenval = np.linalg.eigh(C)[0]
        eignumb = np.linspace(1,len(eigenval),len(eigenval))
        eigenval = eigenval[::-1] #Put largest eigenvals first
        V = np.linalg.eigh(C)[1][:,::-1] # Eigenvectors from covariance matrix with most dominant first
        return C,eignumb,eigenval,V
    ### Remove dominant modes:
    V = np.linalg.eigh(C)[1][:,::-1] # Eigenvectors from covariance matrix with most dominant first
    A = V[:,:N_fg] # Mixing matrix, first N_fg most dominant modes of eigenvectors
    S = np.dot(A.T,M) # not including weights in mode subtraction (as per Yi-Chao's approach)
    Residual = (M - np.dot(A,S))
    Residual = np.reshape(Residual,(nz,nx,ny))
    Residual = np.transpose(Residual,axes=np.argsort(axes))
    if return_A:
        return Residual,A
    return Residual

def check_unit_equiv(u1,u2):
    """
    Check if two units are equivelant
    """
    return ((1*u1/u2).si.unit == units.dimensionless_unscaled)
    