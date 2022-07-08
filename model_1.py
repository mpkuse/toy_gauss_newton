import numpy as np 

""" I define the function and its jacobians """

def f( x, param ):
    return np.exp( param[0]*x ) * np.cos( param[1]*x ) 

def DfDp0( x, param ):
    return param[0] * np.exp( param[0] * x  ) * np.cos( param[1] * x )

def DfDp1( x, param ):
    return np.exp( param[0]*x ) * param[1] * (-np.sin( param[1]*x ) )


