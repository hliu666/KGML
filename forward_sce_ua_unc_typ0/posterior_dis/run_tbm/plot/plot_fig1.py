import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def compute_stats(typ_id, site, final_results, df, variable_name, data):
    final_results[f'min_{variable_name}'] = df.filter(like=variable_name).min(axis=1)
    final_results[f'max_{variable_name}'] = df.filter(like=variable_name).max(axis=1)
    final_results[f'mean_{variable_name}'] = df.filter(like=variable_name).mean(axis=1)
    final_results[f'median_{variable_name}'] = df.filter(like=variable_name).median(axis=1)
    final_results[f'std_{variable_name}'] = df.filter(like=variable_name).std(axis=1)
    final_results[f'2.5pct_{variable_name}'] = df.apply(lambda x: np.percentile(x, 2.5), axis=1)
    final_results[f'97.5pct_{variable_name}'] = df.apply(lambda x: np.percentile(x, 97.5), axis=1)
    final_results[f'5pct_{variable_name}'] = df.apply(lambda x: np.percentile(x, 5), axis=1)
    final_results[f'95pct_{variable_name}'] = df.apply(lambda x: np.percentile(x, 95), axis=1)

    if variable_name == "nee" or variable_name == "gpp":
        final_results['year'] = data['Year']
        final_results['doy'] = data['Doy']
    else:
        final_results['year'] = data['Year']
        final_results['doy'] = data['Doy']
        final_results['hour'] = data['hour']

    # Save the results to a CSV file
    final_results.to_csv(f"../out_merge/{site}_{variable_name}_typ{typ_id}.csv", index=False)

    return final_results


site = "CA-Oas"
root = "C:/Users/liuha/Desktop/TBM_DA/TBM_DAv3"
directory = root + os.sep + "forward_sce_ua_unc_typ0/posterior_dis/run_tbm/out"

site_pd_h = pd.read_csv(os.path.join(root, "flux", site + '.csv'))
site_pd_d = pd.read_csv(os.path.join(root, "flux_d", site + '.csv'))
site_pd_d_nee = site_pd_h.groupby(['year', 'doy'])[['nee', 'nee_unc']].sum().reset_index()

lai_data = []
nee_data = []
lst_data = []
ref_red_data = []
ref_nir_data = []

num_files = 1000
typ_id = 0
for i in range(0, num_files):

    filename_h = f"{i}_CA-Oas_hourly_typ{typ_id}.csv"
    file_path_h = os.path.join(directory, filename_h)

    filename_d = f"{i}_CA-Oas_daily_typ{typ_id}.csv"
    file_path_d = os.path.join(directory, filename_d)

    # if not os.path.exists(file_path_d):
    #     continue

    hourly_data = pd.read_csv(file_path_h)
    daily_data = pd.read_csv(file_path_d)

    daily_nee = hourly_data.groupby(['Year', 'Doy'])['nee'].sum().reset_index()

    # if daily_nee['nee'].max() > 100:
    #    continue

    print(file_path_d)
    lai_data.append(daily_data['lai'].rename(f'lai_{i}'))
    nee_data.append(daily_nee['nee'].rename(f'nee_{i}'))
    lst_data.append(hourly_data['LST'].rename(f'lst_{i}'))
    ref_red_data.append(hourly_data['ref_red'].rename(f'ref_red_{i}'))
    ref_nir_data.append(hourly_data['ref_nir'].rename(f'ref_nir_{i}'))

# Calculate the final statistics across all files for each row
stats_lai = pd.concat(lai_data, axis=1)
stats_nee = pd.concat(nee_data, axis=1)
stats_lst = pd.concat(lst_data, axis=1)
stats_ref_red = pd.concat(ref_red_data, axis=1)
stats_ref_nir = pd.concat(ref_nir_data, axis=1)

results_lai = compute_stats(typ_id, site, pd.DataFrame(), stats_lai, 'lai', daily_data)
results_nee = compute_stats(typ_id, site, pd.DataFrame(), stats_nee, 'nee', daily_nee)
results_lst = compute_stats(typ_id, site, pd.DataFrame(), stats_lst, 'lst', hourly_data)
results_ref_red = compute_stats(typ_id, site, pd.DataFrame(), stats_ref_red, 'ref_red', hourly_data)
results_ref_nir = compute_stats(typ_id, site, pd.DataFrame(), stats_ref_nir, 'ref_nir', hourly_data)

results_lst['obs_lst'] = site_pd_h['lst']
results_lst['obs_lst_unc'] = site_pd_h['lst_unc']
results_lst = results_lst.dropna()

results_ref_red['obs_ref_red'] = site_pd_h['ref_red']
results_ref_red['obs_ref_red_unc'] = site_pd_h['ref_red_unc']
results_ref_red = results_ref_red.dropna()

results_ref_nir['obs_ref_nir'] = site_pd_h['ref_nir']
results_ref_nir['obs_ref_nir_unc'] = site_pd_h['ref_nir_unc']
results_ref_nir = results_ref_nir.dropna()

# Convert 'Doy' and 'Year' to numeric values if they're not already
daily_data['Doy'] = pd.to_numeric(daily_data['Doy'], errors='coerce')
daily_data['Year'] = pd.to_numeric(daily_data['Year'], errors='coerce')
site_pd_d['year'] = pd.to_numeric(site_pd_d['year'], errors='coerce')

# Create a continuous 'Date' column for plotting
start_year = daily_data['Year'].min()
daily_data['Continuous_Doy'] = (daily_data['Year'] - start_year) * 365 + daily_data['Doy']
site_pd_d['Continuous_Doy'] = (site_pd_d['year'] - start_year) * 365 + site_pd_d['doy']
site_pd_d_nee['Continuous_Doy'] = (site_pd_d_nee['year'] - start_year) * 365 + site_pd_d_nee['doy']
site_pd_h['Continuous_Doy'] = (site_pd_h['year'] - start_year) * 365 + site_pd_h['doy']
results_lst['Continuous_Doy'] = (results_lst['year'] - start_year) * 365 + results_lst['doy']
results_ref_red['Continuous_Doy'] = (results_ref_red['year'] - start_year) * 365 + results_ref_red['doy']
results_ref_nir['Continuous_Doy'] = (results_ref_nir['year'] - start_year) * 365 + results_ref_nir['doy']

# Sort the data by 'Continuous_Doy' to ensure it's in the right order for plotting
daily_data = daily_data.sort_values(by='Continuous_Doy')
site_pd_d = site_pd_d.sort_values(by=['year', 'doy'])  # Assuming site_pd_d has similar columns

# Create subplots
# Create subplots with a 3x2 grid
fig, axs = plt.subplots(3, 2, figsize=(14, 20))

# Plot LAI
axs[0, 0].plot(daily_data['Continuous_Doy'], results_lai['median_lai'], color='blue', linewidth=1, label='Median LAI')
axs[0, 0].plot(daily_data['Continuous_Doy'], results_lai['mean_lai'], color='green', linewidth=1, label='Mean LAI')
axs[0, 0].fill_between(daily_data['Continuous_Doy'], results_lai['2.5pct_lai'], results_lai['97.5pct_lai'], color='gray', alpha=0.5, label='Range (2.5-97.5% LAI)')
axs[0, 0].errorbar(site_pd_d['Continuous_Doy'], site_pd_d['lai'], yerr=site_pd_d['lai_std'], fmt='o', color='red', markersize=3, label='Observation with Uncertainty')
axs[0, 0].set_xlabel('Day of Year (Doy)')
axs[0, 0].set_ylabel(f'type {typ_id} Leaf Area Index (LAI)')
axs[0, 0].legend()

# Plot NEE
axs[0, 1].plot(daily_data['Continuous_Doy'], results_nee['median_nee'], color='blue', linewidth=1, label='Median NEE')
axs[0, 1].plot(daily_data['Continuous_Doy'], results_nee['mean_nee'], color='green', linewidth=1, label='Mean NEE')
axs[0, 1].fill_between(daily_data['Continuous_Doy'], results_nee['2.5pct_nee'], results_nee['97.5pct_nee'], color='gray', alpha=0.5, label='Range (2.5-97.5% NEE)')
axs[0, 1].errorbar(site_pd_d_nee['Continuous_Doy'], site_pd_d_nee['nee'], yerr=site_pd_d_nee['nee_unc'], fmt='o', color='red', markersize=3, label='Observation with Uncertainty')
axs[0, 1].set_xlabel('Day of Year (Doy)')
axs[0, 1].set_ylabel(f'type {typ_id} Net Ecosystem Exchange (NEE)')
axs[0, 1].legend()

# Plot LST
axs[1, 0].scatter(results_lst['Continuous_Doy'], results_lst['median_lst'], color='blue', label='Median LST')
axs[1, 0].scatter(results_lst['Continuous_Doy'], results_lst['mean_lst'], color='green', label='Mean LST')
axs[1, 0].fill_between(results_lst['Continuous_Doy'], results_lst['2.5pct_lst'], results_lst['97.5pct_lst'], color='gray', alpha=0.5, label='Range (2.5-97.5% LST)')
axs[1, 0].errorbar(results_lst['Continuous_Doy'], results_lst['obs_lst'], yerr=results_lst['obs_lst_unc'], fmt='o', color='red', markersize=3, label='Observation with Uncertainty')
axs[1, 0].set_xlabel('Day of Year (Doy)')
axs[1, 0].set_ylabel(f'type {typ_id} Land Surface Temperature (LST)')
axs[1, 0].legend()

# Plot Reflectance Red
axs[1, 1].scatter(results_ref_red['Continuous_Doy'], results_ref_red['median_ref_red'], color='blue', label='Median Red Reflectance')
axs[1, 1].scatter(results_ref_red['Continuous_Doy'], results_ref_red['mean_ref_red'], color='green', label='Mean Red Reflectance')
axs[1, 1].fill_between(results_ref_red['Continuous_Doy'], results_ref_red['2.5pct_ref_red'], results_ref_red['97.5pct_ref_red'], color='gray', alpha=0.5, label='Range (2.5-97.5% Red Reflectance)')
axs[1, 1].errorbar(results_ref_red['Continuous_Doy'], results_ref_red['obs_ref_red'], yerr=results_ref_red['obs_ref_red_unc'], fmt='o', color='red', markersize=3, label='Observation with Uncertainty')
axs[1, 1].set_xlabel('Day of Year (Doy)')
axs[1, 1].set_ylabel(f'type {typ_id} Red Reflectance')
axs[1, 1].legend()

# Plot Reflectance NIR
axs[2, 0].scatter(results_ref_nir['Continuous_Doy'], results_ref_nir['median_ref_nir'], color='blue', label='Median NIR Reflectance')
axs[2, 0].scatter(results_ref_nir['Continuous_Doy'], results_ref_nir['mean_ref_nir'], color='green', label='Mean NIR Reflectance')
axs[2, 0].fill_between(results_ref_nir['Continuous_Doy'], results_ref_nir['2.5pct_ref_nir'], results_ref_nir['97.5pct_ref_nir'], color='gray', alpha=0.5, label='Range (2.5-97.5% NIR Reflectance)')
axs[2, 0].errorbar(results_ref_nir['Continuous_Doy'], results_ref_nir['obs_ref_nir'], yerr=results_ref_nir['obs_ref_nir_unc'], fmt='o', color='red', markersize=3, label='Observation with Uncertainty')
axs[2, 0].set_xlabel('Day of Year (Doy)')
axs[2, 0].set_ylabel(f'{typ_id} NIR Reflectance')
axs[2, 0].legend()

# Adjust the unused subplot (2,1)
axs[2, 1].axis('off')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

