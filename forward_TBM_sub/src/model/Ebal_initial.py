# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 14:23:29 2022

@author: hliu
"""
from numpy import exp, radians, cos, sin, pi
import numpy as np

from RTM_initial import weighted_sum_over_lidf_solar_vec, CIxy
from RTM_initial import get_gauss_legendre_quadrature, get_conversion_factors


def calc_extinc_coeff_pars(CI_flag, CI_thres, lidf):
    # Nodes and weights of Gauss-Legendre integrals
    xx, ww = get_gauss_legendre_quadrature()

    conv1_tL, conv2_tL = get_conversion_factors(0.0, np.pi / 2.0)

    neword_tL = conv1_tL * xx + conv2_tL
    mu_tL = np.cos(neword_tL)
    sin_tL = np.sin(neword_tL)

    tta = neword_tL * 180 / pi  # observer zenith angle

    Ga, ka = weighted_sum_over_lidf_solar_vec(tta, lidf)

    CIa = CIxy(CI_flag, tta, CI_thres)

    sum_tL0 = ww * mu_tL * sin_tL * 2 * conv1_tL

    return [ka * CIa, sum_tL0]