import numpy as np
import torch
from carbon_pool import carp_m, calc_nee


def predict(daily_df, hourly_df, ms, vs, dL):
    [rtmo_m, bicm_m] = ms
    [carp_v, rtmo_v, bicm_v] = vs

    rtmo_m.eval()
    bicm_m.eval()

    for i, batch in enumerate(dL.predict_loader):
        carp_X, carp_y, rtmo_X0, rtmo_y0, bicm_X0, bicm_y0 = batch

        if i > 0:
            carp_X[:, 0] = bicm_out_d[:, 0]
            carp_X[:, 3] = clab2
            carp_X[:, 4] = cf2
            carp_X[:, 5] = cr2
            carp_X[:, 6] = cw2
            carp_X[:, 7] = cl2
            carp_X[:, 8] = cs2

        rtmo_X = rtmo_X0.view(-1, rtmo_X0.shape[-1])
        bicm_X = bicm_X0.view(-1, bicm_X0.shape[-1])

        carp_out_d, clab2, cf2, cr2, cw2, cl2, cs2, Theta, theta_lit, theta_som = carp_m(carp_X)
        carp_out_d = torch.tensor(dL.carp_obsrvs_scaler.transform(carp_out_d))
        carp_out_h = carp_out_d.repeat_interleave(24)

        rtmo_X[:, 0] = carp_out_h
        rtmo_out_h = rtmo_m(rtmo_X.float())

        bicm_X[:, 0] = carp_out_h
        bicm_X[:, 5] = rtmo_out_h[:, 0]
        bicm_out_h = bicm_m(bicm_X.float())

        bicm_X_tbm = dL.bicm_frcngs_scaler.inverse_transform(bicm_X[:, 0:len(bicm_v.x_vars)].detach().numpy())  # debug
        bicm_out_h_arr = bicm_out_h.detach().numpy()
        bicm_out_h_scaler = dL.bicm_obsrvs_scaler.inverse_transform(bicm_out_h_arr)

        gpp_hourly = bicm_out_h_scaler[:, 0].reshape(-1, 24)
        ta_hourly = bicm_X_tbm[:, 2].reshape(-1, 24)
        nee_hourly = calc_nee(torch.tensor(gpp_hourly), ta_hourly, cl2, cs2, Theta, theta_lit, theta_som)
        nee_hourly = torch.transpose(nee_hourly, 0, 1)

        # max_value, min_value = dL.carp_input_scaler.data_max_[0], dL.carp_input_scaler.data_min_[0]
        # gpp_scaled_hourly = 2 * ((gpp_hourly - min_value) / (max_value - min_value)) - 1
        gpp_scaled_daily = np.sum(gpp_hourly, axis=1).reshape(-1, 1) * 1.03775 / 24
        bicm_out_d = torch.tensor(gpp_scaled_daily)

        # carp_pred_sub = dL.carp_label_scaler.inverse_transform(carp_out_d.detach().numpy())
        # rtmo_pred_sub = rtmo_out_h.detach().numpy()
        # bicm_pred_sub = dL.bicm_label_scaler.inverse_transform(bicm_out_h.detach().numpy())

        carp_pred_sub = carp_out_d.detach().numpy()
        rtmo_pred_sub = rtmo_out_h.detach().numpy()
        bicm_pred_sub = bicm_out_h.detach().numpy()
        neeh_pred_sub = nee_hourly.detach().numpy()

        if i == 0:
            # a = bicm_X_tbm
            carp_y_pred = carp_pred_sub
            rtmo_y_pred = rtmo_pred_sub
            bicm_y_pred = bicm_pred_sub
            neeh_y_pred = neeh_pred_sub
        else:
            # a = np.vstack((a, bicm_X_tbm))
            carp_y_pred = np.vstack((carp_y_pred, carp_pred_sub))
            rtmo_y_pred = np.vstack((rtmo_y_pred, rtmo_pred_sub))
            bicm_y_pred = np.vstack((bicm_y_pred, bicm_pred_sub))
            neeh_y_pred = np.vstack((neeh_y_pred, neeh_pred_sub))

    carp_pred_scaler = dL.carp_obsrvs_scaler.inverse_transform(carp_y_pred)
    rtmo_pred_scaler = dL.rtmo_obsrvs_scaler.inverse_transform(rtmo_y_pred)
    bicm_pred_scaler = dL.bicm_obsrvs_scaler.inverse_transform(bicm_y_pred)

    daily_df.loc[daily_df.index[:len(carp_pred_scaler)], ['lai']] = carp_pred_scaler
    lai_arr = daily_df['lai'].values

    hourly_df.loc[hourly_df.index[:len(rtmo_pred_scaler)], ['fpar', 'ref_red', 'ref_nir']] = rtmo_pred_scaler
    ref_red_arr = hourly_df['ref_red'].values
    ref_nir_arr = hourly_df['ref_nir'].values

    hourly_df.loc[hourly_df.index[:len(bicm_pred_scaler)], ['an', 'lst']] = bicm_pred_scaler
    lst_arr = hourly_df['lst'].values
    nee_arr = neeh_y_pred

    return lai_arr, nee_arr, lst_arr, ref_red_arr, ref_nir_arr
