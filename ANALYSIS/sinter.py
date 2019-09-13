import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

def K(beta):
    integral_value = quad(
        (lambda epsilon : ((1-epsilon**2)*(1-beta*epsilon**2))**(-1./2.)),
        0,
        1
    )
    return integral_value[0]

def get_t(alpha,mu_to_gamma,alpha_f):
    
    COMPLICATED_INTEGRAL = quad(
                        (lambda beta : (beta*((1+beta)**(1./2.))*K(beta))**-1),
                            alpha**2,
                            0.99
                        )[0]
    
    PREFACTOR = np.pi / 4. * alpha_f * mu_to_gamma
    
    return PREFACTOR * COMPLICATED_INTEGRAL

def get_trajectory(mu_to_gamma, alpha_f, t = 1., delta_t = 0):
    """takes mu_to_gamma, alpha_f, and t as a kwarg, and returns time varying value of y as a fraction of alpha_f"""

    offset_t = t - delta_t 

    alpha_range = np.linspace(0.00001,0.99999,100)

    t_range = [get_t(i,mu_to_gamma,alpha_f) for i in alpha_range]

    ''' 
    print "t range min is %s"%(min(t_range))
    print "t range max is %s"%(max(t_range))
    '''

    t_to_alpha = interp1d(t_range,alpha_range)
    
    y = []

    for t_val in offset_t:
        alpha = t_to_alpha(t_val)
        y_val =  (1-alpha)*(1+alpha**2)**-0.5

        y.append(np.copy(y_val))
        
    return y
