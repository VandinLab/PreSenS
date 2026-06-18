import click
import pandas as pd
from pathlib import Path

from data_preprocessing import LocalDataset, RemoteDataset

def create_snapshots_from_taxi(input_path : Path) -> pd.DataFrame:
    dataset = LocalDataset(name = "taxi")
    data = dataset.fetch(filepath = str(input_path))
    print(f"Shape of data: {data.shape}")
    # drop rows with MISSING_DATA = True column
    data = data[data['MISSING_DATA'] == False]
    print('Shape after removing missing data:', data.shape)
    # convert TIMESTAMP column from unix to datetime, only the date and not the time
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], unit='s').dt.date
    data['MONTH'] = pd.to_datetime(data['TIMESTAMP']).dt.month
    return data


@click.command("Script to create snapshots from data")
@click.option('--input_path', '-i', help='Path to input file', type=click.Path(exists=True), required=False)
@click.option('--name', '-n', help='Datastream name', type=str, required=True)
@click.option('--output_folder', '-o', help='Path to output folder', type=click.Path(exists=True), required=True)
def preprocess_dataset(input_path: Path, name: str, output_folder: Path):

    if name == "taxi":
        data = create_snapshots_from_taxi(input_path)
        unique_months = data['MONTH'].unique()
        for idx_snap, month in enumerate(unique_months):
            current_output_path = output_folder / Path(f'{idx_snap+1:02d}_{name}_snapshot_{month:02d}.csv.gz')
            # take current month snapshot
            snapshot = (data[data['MONTH'] == month]).copy()
            print(f'Snapshot of month {month} shape: {snapshot.shape}')
            # POLYLINE preprocessing
            snapshot['POLYLINE'] = snapshot['POLYLINE'].apply(lambda x: eval(x))
            # check if POLYLINE is empty (and the first element has two locations)
            snapshot = snapshot[snapshot['POLYLINE'].apply(lambda x: (len(x) > 0) and (len(x[0]) == 2))]
            print(f'Snapshot shape after preprocessing: {snapshot.shape}')
            snapshot['start_latitude'] = snapshot['POLYLINE'].apply(lambda x: x[0][0])
            snapshot['start_longitude'] = snapshot['POLYLINE'].apply(lambda x: x[0][1])
            # drop all columns except start_latitude and start_longitude
            snapshot = snapshot[['start_latitude', 'start_longitude']]
            print(f'Writing to output dataset of shape {snapshot.shape} ...')
            snapshot.to_csv(str(current_output_path), index=False, header=False, compression='gzip')

    elif name == "twitter":
        twitter = LocalDataset(name = "twitter")
        data = twitter.fetch(filepath = str(input_path))
        print(data.shape)
        # convert timestamp, that is in format YYYYMMDDHHMMSS -> extract the day
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y%m%d%H%M%S').dt.date
        data['day'] = pd.to_datetime(data['timestamp']).dt.day
        unique_days = data['day'].unique()
        print('Unique days', unique_days)
        for idx_snap, day in enumerate(unique_days):
            current_output_path = output_folder / Path(f'{idx_snap+1:02d}_{name}_snapshot_{day:02d}.csv.gz')
            # take current day snapshot
            snapshot = (data[data['day'] == day]).copy()
            # keep only latitude, longitude and timezone columns
            snapshot = snapshot[['latitude', 'longitude', 'timezone']]
            print(f'Snapshot of day {day} shape: {snapshot.shape}')
            snapshot.to_csv(str(current_output_path), index=False, header=False, compression='gzip')

    elif name == 'IntelLab':
        dataset = LocalDataset(name = "IntelLab")
        X = dataset.fetch(filepath=str(input_path), sep=' ')
        # assign columns to dataframe
        X.columns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']
        print(f"Shape of data: {X.shape}")
        # drop time column
        X = X.drop(columns=['time'])
        # create snapshot for each day
        X['date'] = pd.to_datetime(X['date'], format='%Y-%m-%d')
        unique_dates = X['date'].dt.date.unique()
        for idx_snap, date in enumerate(sorted(unique_dates)):
            current_output_path = output_folder / Path(f'{idx_snap+1:02d}_{name}_snapshot_{date}.csv.gz')
            # take current date snapshot
            snapshot = X[X['date'].dt.date == date].copy()
            # drop date column
            snapshot = snapshot.drop(columns=['date'])
            # remove missing data rows with NaN in any column
            snapshot = snapshot.dropna()
            print(f'{idx_snap+1} # Snapshot of date {date} shape: {snapshot.shape}')
            snapshot.to_csv(str(current_output_path), index=False, header=False, compression='gzip')

    else:
        raise ValueError("Dataset not supported")



if __name__ == '__main__':
    preprocess_dataset()
