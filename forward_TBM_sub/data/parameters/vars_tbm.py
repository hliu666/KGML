import pandas as pd
import os

pft_dict = {"ENF": 1, "DBF": 4, "MF": 3, "OSH": 7, "GRA": 10, "WET": 11, "CRO": 12}


def assign_weights(row, sza_threshold=75, low_weight=0.1, high_weight=2.0):
    """Assign weights based on SZA value."""
    if row['sza'] > sza_threshold:
        return low_weight
    else:
        return high_weight


# Specify your folder path here
folder_path = '../../../flux/'

site_file_path = 'H:/TBM_DA/data/step1.site_select/siteInfo.csv'
site_pd = pd.read_csv(site_file_path, sep=',')

flag = 0
# %% Gapfilling flux Site
for index, row in site_pd.iterrows():
    site_ID = row['Site ID']
    latitude = row['Latitude']
    longitude = row['Longitude']
    site_LC = row['LC']

    # Load the data
    file_path = os.path.join(folder_path, site_ID+".csv")
    data = pd.read_csv(file_path)

    # Columns of interest
    columns_of_interest = ['sw', 'par', 'ta', 'vpd', 'wds', 'lai', 'sza']

    # Remove rows with NaN values in the selected columns
    clean_data = data.dropna(subset=columns_of_interest).copy()
    # Calculate weights for each row
    clean_data['weights'] = clean_data.apply(assign_weights, axis=1)

    # Map months to seasons
    season_mapping = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }

    # Add a new column for season
    clean_data['season'] = clean_data['month'].map(season_mapping)

    # Number of samples per season
    samples_per_season = {
        'Summer': 40,
        'Spring': 25,
        'Autumn': 25,
        'Winter': 5
    }

    # Initialize an empty DataFrame to store the samples
    stratified_sample = pd.DataFrame()

    # Sample from each season stratum
    for season, n_samples in samples_per_season.items():
        season_data = clean_data[clean_data['season'] == season]

        # Check if the stratum is empty
        if season_data.empty:
            print(f"Warning: No data available for {season}. Skipping...")
            continue

        # Check if the stratum has fewer data points than desired sample size
        if len(season_data) < n_samples:
            print(f"Warning: Not enough data for {season}. Adjusting sample size to {len(season_data)}.")
            n_samples = len(season_data)

        season_sample = season_data.sample(n=n_samples, weights='weights', random_state=1)
        stratified_sample = pd.concat([stratified_sample, season_sample])

    stratified_sample['pft'] = pft_dict[site_LC]

    if flag == 0:
        out_df = stratified_sample
        flag = 1
    else:
        out_df = pd.concat([out_df, stratified_sample], axis=0)

out_df = out_df[['pft'] + columns_of_interest]
out_df.to_csv("site_vars.csv", index=False)
