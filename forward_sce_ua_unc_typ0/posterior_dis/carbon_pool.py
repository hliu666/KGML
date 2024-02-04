import numpy as np

radconv = 365.25 / np.pi


def fit_polynomial(ep, mult_fac):
    """ Polynomial used to find phi_f and phi (offset terms used in
    phi_onset and phi_fall), given an evaluation point for the polynomial
    and a multiplication term.
    :param ep: evaluation point
    :param mult_fac: multiplication term
    :return: fitted polynomial value
    """
    cf = [2.359978471e-05, 0.000332730053021, 0.000901865258885,
          -0.005437736864888, -0.020836027517787, 0.126972018064287,
          -0.188459767342504]
    poly_val = cf[0] * ep ** 6 + cf[1] * ep ** 5 + cf[2] * ep ** 4 + cf[3] * ep ** 3 + cf[4] * ep ** 2 + \
               cf[5] * ep ** 1 + cf[6] * ep ** 0
    phi = poly_val * mult_fac
    return phi


def phi_onset(doy, d_onset, cronset):
    """Leaf onset function (controls labile to foliar carbon transfer)
    takes d_onset value, cronset value and returns a value for phi_onset.
    """
    release_coeff = np.sqrt(2.) * cronset / 2.
    mag_coeff = (np.log(1. + 1e-3) - np.log(1e-3)) / 2.
    offset = fit_polynomial(1 + 1e-3, release_coeff)
    phi_onset = (2. / np.sqrt(np.pi)) * (mag_coeff / release_coeff) * \
                np.exp(-(np.sin((doy - d_onset + offset) / radconv) * (radconv / release_coeff)) ** 2)
    return phi_onset


def phi_fall(doy, d_fall, crfall, clspan):
    """Leaf fall function (controls foliar to litter carbon transfer) takes
    d_fall value, crfall value, clspan value and returns a value for phi_fall.
    """
    release_coeff = np.sqrt(2.) * crfall / 2.
    mag_coeff = (np.log(clspan) - np.log(clspan - 1.)) / 2.
    offset = fit_polynomial(clspan, release_coeff)
    phi_fall = (2. / np.sqrt(np.pi)) * (mag_coeff / release_coeff) * \
               np.exp(-(np.sin((doy - d_fall + offset) / radconv) * radconv / release_coeff) ** 2)
    return phi_fall


def temp_term(Theta, temperature):
    """ Calculates the temperature exponent factor for carbon pool
    respiration's given a value for Theta parameter.
    :param Theta: temperature dependence exponent factor
    :return: temperature exponent respiration
    """
    temp_term = np.exp(Theta * temperature)
    return temp_term


def carp_m(carp_X):
    gpp, doy, ta,\
    clab, cf, cr, cw, cl, cs, \
    clspan, lma, f_auto, f_fol, f_lab, \
    Theta, theta_min, theta_woo, theta_roo, theta_lit, theta_som,\
    d_onset, cronset, d_fall, crfall = carp_X.flatten()

    f_roo = 1 - f_fol - f_lab

    temp = temp_term(Theta, ta)
    phi_on = phi_onset(doy, d_onset, cronset)
    phi_off = phi_fall(doy, d_fall, crfall, clspan)

    clab2 = (1 - phi_on) * clab + (1 - f_auto) * (1 - f_fol) * f_lab * gpp
    if doy > (d_fall + crfall / 2.0):
        clab2 += cf
        cf2 = 0.0
    else:
        cf2 = (1 - phi_off) * cf + phi_on * clab + (1 - f_auto) * f_fol * gpp

    cr2 = (1 - theta_roo) * cr + (1 - f_auto) * (1 - f_fol) * (1 - f_lab) * f_roo * gpp
    cw2 = (1 - theta_woo) * cw + (1 - f_auto) * (1 - f_fol) * (1 - f_lab) * (1 - f_roo) * gpp
    cl2 = (1 - (theta_lit + theta_min) * temp) * cl + theta_roo * cr + phi_off * cf
    cs2 = (1 - theta_som * temp) * cs + theta_woo * cw + theta_min * temp * cl

    return np.array([[cf2 / lma]]), clab2, cf2, cr2, cw2, cl2, cs2, Theta, theta_lit, theta_som


def calc_nee(gpp, ta, cl, cs, Theta, theta_lit, theta_som):
    nee = -gpp + (theta_lit * cl + theta_som * cs) * temp_term(Theta, ta)
    return nee
