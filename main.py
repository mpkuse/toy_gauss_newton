import numpy as np
import matplotlib.pyplot as plt

from gen_data import gen_data, gen_plotting_data
from model_1 import f, DfDp0, DfDp1

import code 

np.random.seed(10)


def plot_for_params( param, color, label='' ):
    x_hidden, y_hidden = gen_plotting_data( f, param[0], param[1], 0, 10 )
    plt.plot(x_hidden, y_hidden, c =color, label=label )


#---- Known parameters of the data generator ----#
a_hidden = -0.1
b_hidden = 3.0



#---- Initial guess of params ----#
init_guess_params = np.array( [-0.15, 2.7] )


#--- Generate Observation Data ---#
x_obs, y_obs = gen_data(f, a_hidden, b_hidden, 0, 10, 35 )
#^^ x_obs : Nx1 ; y_obs : Nx1



#---- Plot ----#
plt.scatter(x_obs, y_obs, c ="blue", label="observation points")




plot_for_params( init_guess_params, 'green', 'initial guess' )
plot_for_params( (a_hidden,b_hidden), 'red', 'GT' )


#--- Gauss-Newton Method ---#
xparam = init_guess_params
gamma = 0.08

if True:
    prev_error = 1E6
    prev_xparam = np.array( [] )
    prev_J = np.array( [] )
    prev_r = np.array( [] )
    for itr in range(0,35):
        print( "---itr#%d. xparam=%s"%( itr, xparam ) )

        _DfDp0 = DfDp0( x_obs, xparam) #Nx1
        _DfDp1 = DfDp1( x_obs, xparam) #Nx1
        J = np.transpose( np.vstack( ( _DfDp0, _DfDp1 )  ) ) #Nx2
        # print( J.shape, _DfDp0.shape , _DfDp1.shape )

        # Normalize Jacobian 
        J[:,0] = J[:,0] / sum(J[:,0])
        J[:,1] = J[:,1] / sum(J[:,1])

        r = f( x_obs, xparam ) - y_obs
        error = np.dot( r,r )
        print( 'total residue = ', error, ' error_change=', abs(error - prev_error))

        did_we_backtrack = False 
        if error > prev_error:
            xparam = prev_xparam
            error = prev_error
            J = prev_J 
            r = prev_r 
            gamma = gamma * 0.75
            did_we_backtrack = True 
            print( "**Error increase, backtrack to previous estimate=", prev_xparam, ' new_gamma=', gamma ) 

              

        if did_we_backtrack == False and abs(error - prev_error) < 1E-5:
            print( "**Optimization Converged")
            break 

        # Normal Eq
        H = np.matmul(  np.transpose(J), J )
        b = -np.matmul(  np.transpose(J), r )
        
        # print( "H=\n", H )
        # print( "b=", b )

        # Solve normal eq 
        delta = np.matmul( np.linalg.inv(H) , b )
        # delta = np.linalg.lstsq(H, b)
        # delta = b
        # delta = np.matmul( np.linalg.inv(H) , b ) + 0.1*b

        # Update estimate 
        prev_xparam = xparam
        xparam = xparam - gamma*delta

        # Book keeping 
        prev_error = error
        prev_J = J 
        prev_r = r

plot_for_params( xparam, 'cyan', 'after GN' )


plt.legend()
plt.show()