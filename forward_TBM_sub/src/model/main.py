import joblib
import pandas as pd
import numpy as np
import scipy.io
import sys

from data import TBM_Data
from parameter import TBM_Pars
from model import TBM_Model


def main():
    rsr_red = np.genfromtxt(sys.argv[1])  # "src/model/support/rsr_red.txt"
    rsr_nir = np.genfromtxt(sys.argv[2])  # "src/model/support/rsr_nir.txt"
    rsr_sw1 = np.genfromtxt(sys.argv[3])  # "src/model/support/rsr_swir1.txt"
    rsr_sw2 = np.genfromtxt(sys.argv[4])  # "src/model/support/rsr_swir2.txt"
    prospectpro = np.loadtxt(sys.argv[5])  # "src/model/support/dataSpec_PDB.txt"
    soil = np.genfromtxt(sys.argv[6])  # "src/model/support/soil_reflectance.txt"
    TOCirr = np.loadtxt(sys.argv[7], skiprows=1)  # "src/model/support/atmo.txt"

    """
    rsr_red = np.genfromtxt("support/rsr_red.txt")  #
    rsr_nir = np.genfromtxt("support/rsr_nir.txt")  #
    rsr_sw1 = np.genfromtxt("support/rsr_swir1.txt")  #
    rsr_sw2 = np.genfromtxt("support/rsr_swir2.txt")  #
    prospectpro = np.loadtxt("support/dataSpec_PDB.txt")  #
    soil = np.genfromtxt("support/soil_reflectance.txt")  #
    TOCirr = np.loadtxt("support/atmo.txt", skiprows=1)  #
    """

    driving_data = [rsr_red, rsr_nir, rsr_sw1, rsr_sw2, prospectpro, soil, TOCirr]

    i, js = int(sys.argv[11]), int(sys.argv[12])
    # i, js = 0, 2370

    nvars = 10

    """
    # ...debug...
    # xh = 3035
    # self.d.sw[xh],self.d.par[xh],self.d.t_mean[xh], self.d.vpd[xh], self.d.precip[xh], self.d.wds[xh],lai, self.d.tts[xh]
    forcing = np.array([1,
                        642.88,
                        1298.585,
                        17.53981457,
                        694.9888477,
                        0.0,
                        2.336977473,
                        1.3628462412316165,
                        28.674167720699337,
                        0,
                        149.62765802340328])

    # self.p.CI_thres, self.p.lidfa, self.p.Cab, self.p.Cm,self.p.RUB, self.p.Rdsc, self.p.CB6F, self.p.gm, self.p.BallBerrySlope, self.p.BallBerry0
    params = np.array([0.7625,
                       39.375,
                       21.875,
                       0.0099375 * 10000.0,
                       124.21875,
                       0.00084375,
                       77.34375,
                       3.9084375,
                       8.90625,
                       0.78125])

    p = TBM_Pars(params)
    d = TBM_Data(p, forcing, driving_data)
    m = TBM_Model(d, p)
    
    rtmo_out_sub, bicm_out_sub = m.tbm()
    """

    for j in range(0, js):
        k = i + j

        inputs = np.load(sys.argv[10])[k]
        # inputs = np.load("../../data/parameters/input_data_test.npy")[k]

        forcing, params = inputs[0:nvars], inputs[nvars:]

        p = TBM_Pars(params)
        d = TBM_Data(p, forcing, driving_data)
        m = TBM_Model(d, p)

        tbm_out_sub = m.tbm()
        out_sub = np.concatenate((forcing, params, tbm_out_sub))

        if j == 0:
            out = out_sub
        else:
            out = np.vstack((out, out_sub))

    out_df = pd.DataFrame(out,
                          columns=['pft', 'sw', 'par', 'ta',
                                   'vpd', 'wds', 'lai',
                                   'sza', 'vza', 'raa',
                                   "CI", "lidf", "cab", "lma",
                                   "RUB", "Rdsc", "CB6F", "gm", "BallBerrySlope", "BallBerry0",
                                   'an', 'lst', 'fpar', 'ref_red', 'ref_nir', 'brf_red', 'brf_nir'
                                   ])

    out_df.to_csv('{0}_forward_hourly.csv'.format(i // js), index=False)


if __name__ == '__main__':
    main()
