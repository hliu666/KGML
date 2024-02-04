# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:44:43 2022

@author: hliu
"""
from numpy import pi
import numpy as np
from tbm.RTM_initial import CIxy, weighted_sum_over_lidf_solar


def rtm_o(d, p, x, lai, hemi_dif_brf):
    rho, tau = d.leaf[0][:, x % 365], d.leaf[1][:, x % 365]
    rg = d.soil
    lidf = d.lidf

    CI_flag = p.CI_flag
    CI_thres = p.CI_thres
    CIs, CIo = d.CIs[x], d.CIo[x]
    ks, ko = d.ks[x], d.ko[x]
    sob, sof = d.sob[x], d.sof[x]
    tts, tto, psi = d.tts[x], d.tto[x], d.psi[x]

    # soil and canopy properties
    w = rho + tau  # leaf single scattering albedo

    i0 = max(1 - np.exp(-ks * lai * CIs), 1e-16)
    iv = max(1 - np.exp(-ko * lai * CIo), 1e-16)

    t0 = 1 - i0
    tv = 1 - iv

    dso = define_geometric_constant(tts, tto, psi)
    if np.isscalar(dso):
        [kc, kg] = hotspot_calculations(lai, ko, ks, CIo, CIs, dso)
    else:
        [kc, kg] = hotspot_calculations_vec(lai, ko, ks, CIo, CIs, dso)

    [sob_vsla, sof_vsla, kgd] = BRF_hemi_func(d.hemi_pars, lai, x)

    [sob_vsla_dif, sof_vsla_dif, kg_dif] = BRF_dif_func(d.dif_pars, lai, x)

    [sob_vsla_hemi_dif, sof_vsla_hemi_dif, kgd_dif] = hemi_dif_brf

    rho2 = iv / 2 / lai

    iD = i_hemi(CI_flag, lai, lidf, CI_thres)
    td = 1 - iD

    p = 1 - iD / lai

    rho_hemi = iD / 2 / lai
    rho_dif = iv / 2 / lai
    rho_dif_hemi = iD / 2 / lai

    wso = sob * rho + sof * tau

    Tdn = t0 + i0 * w * rho_hemi / (1 - p * w)
    Tup_o = tv + iD * w * rho2 / (1 - p * w)
    Rdn = iD * w * rho_hemi / (1 - p * w)

    BRFv = wso * kc / ko + i0 * w * w * p * rho2 / (1 - p * w)
    BRFs = kg * rg
    BRFm = rg * Tdn * Tup_o / (1 - rg * Rdn) - t0 * rg * tv
    BRF = BRFv + BRFs + BRFm

    Tup_hemi = td + iD * w * rho_hemi / (1 - p * w)

    Rv = sob_vsla * rho + sof_vsla * tau + i0 * w ** 2 * p * rho_hemi / (1 - p * w)
    Rs = kgd * rg
    Rm = rg * Tdn * Tup_hemi / (1 - rg * Rdn) - t0 * rg * td
    R = Rv + Rs + Rm  # albedo

    # absorption
    Av = i0 * (1 - w) / (1 - p * w)
    Aup = iD * (1 - w) / (1 - p * w)
    Am = rg * (Tdn) * (Aup) / (1 - rg * (Rdn))
    A_tot = Av + Am  # absorption

    Tdn_dif = td + iD * w * rho_dif_hemi / (1 - p * w)
    Tup_difo = tv + iD * w * rho_dif / (1 - p * w)
    Rdn_dif = iD * w * rho_dif_hemi / (1 - p * w)

    BRF_difv = sob_vsla_dif * rho + sof_vsla_dif * tau + iD * w ** 2 * p * rho_dif / (1 - p * w)
    BRF_difs = kg_dif * rg
    BRF_difm = rg * (Tdn_dif) * (Tup_difo) / (1 - rg * (Rdn_dif)) - td * rg * tv
    BRF_dif = BRF_difv + BRF_difs + BRF_difm

    Tup_dif_hemi = td + iD * w * rho_dif_hemi / (1 - p * w)

    R_difv = sob_vsla_hemi_dif * rho + sof_vsla_hemi_dif * tau + iD * w ** 2 * p * rho_dif_hemi / (1 - p * w)
    R_difs = kgd_dif * rg
    R_difm = rg * (Tdn_dif) * (Tup_dif_hemi) / (1 - rg * (Rdn_dif)) - td * rg * td
    R_dif = R_difv + R_difs + R_difm

    # absorption
    Aup_dif = iD * (1 - w) / (1 - p * w)
    A_difv = iD * (1 - w) / (1 - p * w)
    A_difm = rg * (Tdn_dif) * (Aup_dif) / (1 - rg * (Rdn_dif))
    A_dif = A_difv + A_difm

    fPAR = sum(A_tot[0:301]) / 301

    sur_refl_b01 = float(np.nansum(BRF[220:271].flatten() * d.rsr_red.flatten()) / np.nansum(d.rsr_red.flatten()))
    sur_refl_b02 = float(np.nansum(BRF[441:477].flatten() * d.rsr_nir.flatten()) / np.nansum(d.rsr_nir.flatten()))

    if tts > 75 or lai == 0:
        fPAR = 0.0
        sur_refl_b01 = 0.0
        sur_refl_b02 = 0.0

    out = {'fPAR': fPAR,
           'BRF': BRF,
           'kc': kc,
           'kg': kg,
           'i0': i0,
           'iD': iD,
           'A_tot': A_tot,
           'A_dif': A_dif,
           'R': R,
           'R_dif': R_dif,
           'Rs': Rs,
           'p': p,
           'rho2': rho2,
           'rho_hemi': rho_hemi,
           'tv': tv,
           'sob_vsla': sob_vsla,
           'sof_vsla': sof_vsla,
           'kgd': kgd,
           'sob_vsla_dif': sob_vsla_dif,
           'sof_vsla_dif': sof_vsla_dif,
           'kg_dif': kg_dif,
           'sur_refl_red': sur_refl_b01,
           'sur_refl_nir': sur_refl_b02
           }

    return out


def hotspot_calculations(lai, ko, ks, CIo, CIs, dso):
    ko_1 = ko * CIo
    ks_1 = ks * CIs

    # Treatment of the hotspot-effect
    alf = 1e36

    hotspot = 0.05

    tss = np.exp(-ks_1 * lai)

    # Apply correction 2/(K+k) suggested by F.-M. Breon
    if hotspot > 0.:
        alf = (dso / hotspot) * 2. / (ks_1 + ko_1)
    if alf == 0.:
        # The pure hotspot
        tsstoo = tss
        sumint = (1. - tss) / (ks_1 * lai)
    else:
        # Outside the hotspot
        alf = (dso / hotspot) * 2. / (ks_1 + ko_1)
        fhot = lai * np.sqrt(ko_1 * ks_1)
        # Integrate by exponential Simpson method in 20 steps the steps are arranged according to equal partitioning of the slope of the joint probability function
        x1 = 0.
        y1 = 0.
        f1 = 1.
        fint = (1. - np.exp(-alf)) * .05
        sumint = 0.
        for istep in range(1, 21):
            if istep < 20:
                x2 = -np.log(1. - istep * fint) / alf
            else:
                x2 = 1.
            y2 = -(ko_1 + ks_1) * lai * x2 + fhot * (1. - np.exp(-alf * x2)) / alf
            f2 = np.exp(y2)
            sumint = sumint + (f2 - f1) * (x2 - x1) / (y2 - y1)
            x1 = x2
            y1 = y2
            f1 = f2

        tsstoo = f1
        if np.isnan(sumint):
            sumint = 0.

    gammasos = ko_1 * lai * sumint
    # gammasos = max(gammasos, 1e-16)
    # tsstoo = max(tsstoo, 1e-16)
    return gammasos, tsstoo  # kc, kg


def hotspot_calculations_vec(lai, ko, ks, CIo, CIs, dso):
    ko_1 = ko * CIo
    ks_1 = ks * CIs

    hotspot = np.full(ko_1.shape, 0.05)

    tss = np.exp(-ks_1 * lai)

    tsstoo = np.zeros(tss.shape)
    sumint = np.zeros(lai.shape)

    # Treatment of the hotspot-effect
    alf = np.ones(lai.shape) * 1e36
    alf[hotspot > 0] = (dso[hotspot > 0] / hotspot[hotspot > 0]) * 2. / (ks_1[hotspot > 0] + ko_1[hotspot > 0])

    index = np.logical_and(lai > 0, alf == 0)
    # The pure hotspot
    tsstoo[index] = tss[index]
    sumint[index] = (1. - tss[index]) / (ks_1[index] * lai[index])

    # Outside the hotspot
    index = np.logical_and(lai > 0, alf != 0)
    fhot = lai[index] * np.sqrt(ko_1[index] * ks_1[index])
    # Integrate by exponential Simpson method in 20 steps the steps are arranged according to equal partitioning of the slope of the joint probability function
    x1 = np.zeros(fhot.shape)
    y1 = np.zeros(fhot.shape)
    f1 = np.ones(fhot.shape)
    fint = (1. - np.exp(-alf[index])) * .05
    for istep in range(1, 21):
        if istep < 20:
            x2 = -np.log(1. - istep * fint) / alf[index]
        else:
            x2 = np.ones(fhot.shape)
        y2 = -(ko_1[index] + ks_1[index]) * lai[index] * x2 + fhot * (1. - np.exp(-alf[index] * x2)) / alf[index]
        f2 = np.exp(y2)
        sumint[index] = sumint[index] + (f2 - f1) * (x2 - x1) / (y2 - y1)
        x1 = np.copy(x2)
        y1 = np.copy(y2)
        f1 = np.copy(f2)

    tsstoo[index] = f1
    sumint[np.isnan(sumint)] = 0.
    # return tsstoo, sumint
    gammasos = ko_1 * lai * sumint
    return gammasos, tsstoo  # kc, kg


def define_geometric_constant(tts, tto, psi):
    tants = np.tan(np.radians(tts))
    tanto = np.tan(np.radians(tto))
    cospsi = np.cos(np.radians(psi))
    dso = np.sqrt(tants ** 2. + tanto ** 2. - 2. * tants * tanto * cospsi)
    return dso


def BRF_hemi_func(pars, lai, x):
    xx = np.array([0.9602898565, -0.9602898565, 0.7966664774, -0.7966664774, 0.5255324099, -0.5255324099, 0.1834346425,
                   -0.1834346425])

    ww = np.array([0.1012285363, 0.1012285363, 0.2223810345, 0.2223810345, 0.3137066459, 0.3137066459, 0.3626837834,
                   0.3626837834])

    # * define limits of integration and the convertion factors for integration
    # * over thetaL (note the tL suffix!)
    upperlimit_tL = np.pi / 2.0
    lowerlimit_tL = 0.0
    conv1_tL = (upperlimit_tL - lowerlimit_tL) / 2.0
    conv2_tL = (upperlimit_tL + lowerlimit_tL) / 2.0
    neword_tL = conv1_tL * xx + conv2_tL

    # * define limits of integration and the convertion factors for integration
    # * over phiL (note the pL suffix!)
    upperlimit_pL = 2.0 * pi
    lowerlimit_pL = 0.0
    conv1_pL = (upperlimit_pL - lowerlimit_pL) / 2.0
    conv2_pL = (upperlimit_pL + lowerlimit_pL) / 2.0
    neword_pL = conv1_pL * xx + conv2_pL

    [tts, tto, psi, ks, ko, sob, sof, CIs, CIo] = pars
    tts, tto, psi = tts[x * 64:(x + 1) * 64], tto[x * 64:(x + 1) * 64], psi[x * 64:(x + 1) * 64]
    dso = define_geometric_constant(tts, tto, psi)
    ks, ko = ks[x * 64:(x + 1) * 64], ko[x * 64:(x + 1) * 64]
    sob, sof = sob[x * 64:(x + 1) * 64], sof[x * 64:(x + 1) * 64]
    CIs, CIo = CIs[x * 64:(x + 1) * 64], CIo[x * 64:(x + 1) * 64]

    lai = np.full(tts.shape, lai)
    kca, kga = hotspot_calculations_vec(lai, ko, ks, CIo, CIs, dso)

    k1 = (sob * kca / ko / pi).reshape(8, 8)
    k2 = (sof * kca / ko / pi).reshape(8, 8)
    k3 = (kga / pi).reshape(8, 8)

    neword_tL = conv1_tL * xx + conv2_tL
    mu_tL = np.cos(neword_tL)
    sin_tL = np.sin(neword_tL)

    ww1 = ww * conv1_pL * mu_tL * sin_tL
    ww2 = ww * conv1_tL

    sob_vsla = np.einsum('ij,i,j->', k1, ww1, ww2)
    sof_vsla = np.einsum('ij,i,j->', k2, ww1, ww2)
    kgd = np.einsum('ij,i,j->', k3, ww1, ww2)

    return sob_vsla, sof_vsla, kgd


def BRF_dif_func(pars, lai, x):
    xx = np.array([0.9602898565, -0.9602898565, 0.7966664774, -0.7966664774, 0.5255324099, -0.5255324099, 0.1834346425,
                   -0.1834346425])

    ww = np.array([0.1012285363, 0.1012285363, 0.2223810345, 0.2223810345, 0.3137066459, 0.3137066459, 0.3626837834,
                   0.3626837834])

    # * define limits of integration and the convertion factors for integration
    # * over thetaL (note the tL suffix!)
    upperlimit_tL = np.pi / 2.0
    lowerlimit_tL = 0.0
    conv1_tL = (upperlimit_tL - lowerlimit_tL) / 2.0
    conv2_tL = (upperlimit_tL + lowerlimit_tL) / 2.0

    # * define limits of integration and the convertion factors for integration
    # * over phiL (note the pL suffix!)
    upperlimit_pL = 2.0 * pi
    lowerlimit_pL = 0.0
    conv1_pL = (upperlimit_pL - lowerlimit_pL) / 2.0
    conv2_pL = (upperlimit_pL + lowerlimit_pL) / 2.0

    [tta, tto, psi, ks, ko, sob, sof, CIs, CIo] = pars
    tta, tto, psi = tta[x * 64:(x + 1) * 64], tto[x * 64:(x + 1) * 64], psi[x * 64:(x + 1) * 64]
    dso = define_geometric_constant(tta, tto, psi)
    ks, ko = ks[x * 64:(x + 1) * 64], ko[x * 64:(x + 1) * 64]
    sob, sof = sob[x * 64:(x + 1) * 64], sof[x * 64:(x + 1) * 64]
    CIs, CIo = CIs[x * 64:(x + 1) * 64], CIo[x * 64:(x + 1) * 64]

    lai = np.full(tta.shape, lai)
    kca, kga = hotspot_calculations_vec(lai, ko, ks, CIo, CIs, dso)

    k1 = (sob * kca / ko / pi).reshape(8, 8)
    k2 = (sof * kca / ko / pi).reshape(8, 8)
    k3 = (kga / pi).reshape(8, 8)

    neword_tL = conv1_tL * xx + conv2_tL
    mu_tL = np.cos(neword_tL)
    sin_tL = np.sin(neword_tL)

    ww1 = ww * conv1_pL * mu_tL * sin_tL
    ww2 = ww * conv1_tL

    sob_vsla = np.einsum('ij,i,j->', k1, ww1, ww2)
    sof_vsla = np.einsum('ij,i,j->', k2, ww1, ww2)
    kgd = np.einsum('ij,i,j->', k3, ww1, ww2)

    return sob_vsla, sof_vsla, kgd


def BRF_hemi_dif_func(pars, lai):
    xx = np.array([0.9602898565, -0.9602898565, 0.7966664774, -0.7966664774, 0.5255324099, -0.5255324099, 0.1834346425,
                   -0.1834346425])

    ww = np.array([0.1012285363, 0.1012285363, 0.2223810345, 0.2223810345, 0.3137066459, 0.3137066459, 0.3626837834,
                   0.3626837834])

    # * define limits of integration and the convertion factors for integration
    # * over thetaL (note the tL suffix!)
    upperlimit_mL = pi / 2.0
    lowerlimit_mL = 0.0
    conv1_mL = (upperlimit_mL - lowerlimit_mL) / 2.0
    conv2_mL = (upperlimit_mL + lowerlimit_mL) / 2.0

    #   * define limits of integration and the convertion factors for integration
    # * over phiL (note the pL suffix!)
    upperlimit_nL = 2.0 * pi
    lowerlimit_nL = 0.0
    conv1_nL = (upperlimit_nL - lowerlimit_nL) / 2.0
    conv2_nL = (upperlimit_nL + lowerlimit_nL) / 2.0

    # * define limits of integration and the convertion factors for integration
    # * over thetaL (note the tL suffix!)
    upperlimit_tL = pi / 2.0
    lowerlimit_tL = 0.0
    conv1_tL = (upperlimit_tL - lowerlimit_tL) / 2.0
    conv2_tL = (upperlimit_tL + lowerlimit_tL) / 2.0

    # * define limits of integration and the convertion factors for integration
    # * over phiL (note the pL suffix!)
    upperlimit_pL = 2.0 * pi
    lowerlimit_pL = 0.0
    conv1_pL = (upperlimit_pL - lowerlimit_pL) / 2.0
    conv2_pL = (upperlimit_pL + lowerlimit_pL) / 2.0

    [tts, tto, psi, ks, ko, sob, sof, CIs, CIo] = pars
    dso = define_geometric_constant(tts, tto, psi)

    lai = np.full(tts.shape, lai)
    kca, kga = hotspot_calculations_vec(lai, ko, ks, CIo, CIs, dso)

    k1 = (sob * kca / ko / pi).reshape(8, 8, 8, 8)
    k2 = (sof * kca / ko / pi).reshape(8, 8, 8, 8)
    k3 = (kga / pi).reshape(8, 8, 8, 8)

    neword_tL = conv1_tL * xx + conv2_tL
    mu_tL = np.cos(neword_tL)
    sin_tL = np.sin(neword_tL)

    neword_mL = conv1_mL * xx + conv2_mL
    mu_mL = np.cos(neword_mL)
    sin_mL = np.sin(neword_mL)

    ww1 = ww * conv1_pL * mu_tL * sin_tL
    ww2 = ww * conv1_tL / pi
    ww3 = ww * conv1_nL * mu_mL * sin_mL
    ww4 = ww * conv1_mL

    sob_vsla = np.einsum('ijkl,i,j,k,l->', k1, ww1, ww2, ww3, ww4)
    sof_vsla = np.einsum('ijkl,i,j,k,l->', k2, ww1, ww2, ww3, ww4)
    kgd_dif = np.einsum('ijkl,i,j,k,l->', k3, ww1, ww2, ww3, ww4)

    return sob_vsla, sof_vsla, kgd_dif


def i_hemi(CI_flag, lai, lidf, CI_thres):
    xx = np.array([0.9602898565, -0.9602898565, 0.7966664774, -0.7966664774, 0.5255324099, -0.5255324099, 0.1834346425,
                   -0.1834346425])

    ww = np.array([0.1012285363, 0.1012285363, 0.2223810345, 0.2223810345, 0.3137066459, 0.3137066459, 0.3626837834,
                   0.3626837834])

    # * define limits of integration and the convertion factors for integration
    # * over thetaL (note the tL suffix!)
    upperlimit_tL = np.pi / 2.0
    lowerlimit_tL = 0.0
    conv1_tL = (upperlimit_tL - lowerlimit_tL) / 2.0
    conv2_tL = (upperlimit_tL + lowerlimit_tL) / 2.0

    sum_tL = 0

    for i in range(len(ww)):
        neword_tL = conv1_tL * xx[i] + conv2_tL
        mu_tL = np.cos(neword_tL)
        sin_tL = np.sin(neword_tL)

        tta = neword_tL * 180 / pi  # observer zenith angle

        Ga, ka = weighted_sum_over_lidf_solar(tta, lidf)

        CIa = CIxy(CI_flag, tta, CI_thres)

        ia = 1 - np.exp(-ka * lai * CIa)

        sum_tL = sum_tL + ww[i] * mu_tL * sin_tL * ia * 2

    sum_tL = sum_tL * conv1_tL
    return sum_tL
