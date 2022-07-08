import numpy as np 
import code 


def gen_data( f, a, b, x_min, x_max, n_samples=10 ):
    """
    Given hidden params a,b, will generate data 
    with x drawn randomly from uniform_dist( x_min, y_min )
    y = e^(ax) * cos( b*x )
    """

    x = np.random.uniform(x_min, x_max, n_samples )
    y = f( x, (a,b) )
    return x, y


def gen_plotting_data( f, a, b, x_min, x_max ):

    x = np.linspace( x_min, x_max, num=100)
    y = f( x, (a,b) )
    return x, y


