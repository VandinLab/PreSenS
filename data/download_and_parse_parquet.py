import pyarrow.parquet as pq
import click
import os
import pandas as pd
import numpy as np

@click.command("Script to download NYC snapshots and parse Parquet files")
@click.option("-o", "--output_root", help="Path to output folder CSV file", type=click.Path(), required=True)
def download_and_parse_parquet(output_root: str) -> None:

    start_year = 2015
    end_year = 2024

    # First pass - identify all numeric columns across years and months
    print("=" * 60)
    print("Pass 1: Identifying all numeric columns across all years...")
    print("=" * 60)

    all_numeric_columns = set()
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            month_str = f"{month:02d}"
            url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month_str}.parquet"
            input_path = os.path.join(output_root, f"yellow_tripdata_{year}-{month_str}.parquet")

            if not os.path.exists(input_path):
                print(f"Downloading {url} ...")
                os.system(f"wget {url} -O {input_path}")

            # Read just the schema to identify columns
            # try catch in case download is forbidden (which happens for 2023-10)
            try:
                table = pq.read_table(input_path)
                df = table.to_pandas()
                numeric_features = set(df.select_dtypes(include=[np.number]).columns.tolist())
                if len(all_numeric_columns) == 0:
                    all_numeric_columns = numeric_features
                else:
                    all_numeric_columns = all_numeric_columns.intersection(numeric_features)
                print(f"  {year}-{month_str}: Found {len(numeric_features)} numeric columns")
            except Exception as e:
                print(f"  {year}-{month_str}: Failed to read Parquet file. Error: {e}")

    # Convert to sorted list for consistent ordering
    all_numeric_columns = sorted(list(all_numeric_columns))
    print(f"\n-> Total unique numeric columns across all data: {len(all_numeric_columns)}")
    print(f"  Columns: {all_numeric_columns}")

    # Second pass - aggregate data with consistent columns
    print("\n" + "=" * 60)
    print("Pass 2: Aggregating data with consistent columns...")
    print("=" * 60)

    for year in range(start_year, end_year + 1):
        yearly_aggregated_df = pd.DataFrame()
        output_path_year = os.path.join(output_root, f"yellow_tripdata_{year}_numeric.csv")

        if os.path.exists(output_path_year):
            print(f"Yearly aggregated file for {year} already exists, skipping...")
            continue

        for month in range(1, 13):
            month_str = f"{month:02d}"
            input_path = os.path.join(output_root, f"yellow_tripdata_{year}-{month_str}.parquet")

            # Read the Parquet file
            print(f"Reading {input_path} ...")
            table = pq.read_table(input_path)
            df = table.to_pandas()
            print(f"  Shape: {df.shape}")

            # Retain only the identified numeric columns
            df = df[all_numeric_columns]
            print(f"  Shape after retaining numeric columns: {df.shape}")
            # Drop rows with any missing values
            df = df.dropna()
            print(f"  Shape after dropping NA: {df.shape}")
            # Append to yearly aggregated dataframe
            yearly_aggregated_df = pd.concat([yearly_aggregated_df, df], ignore_index=True)

            # Remove the Parquet file to save space
            # os.remove(input_path)

        # Save the aggregated dataframe to CSV
        print(f'--> Shape of aggregated data for {year}: {yearly_aggregated_df.shape}')

        # Final dropna (should be redundant but ensures clean data)
        yearly_aggregated_df = yearly_aggregated_df.dropna()
        print(f'--> Shape after final cleaning for {year}: {yearly_aggregated_df.shape}')

        # Write to CSV without header
        yearly_aggregated_df.to_csv(output_path_year, index=False, header=False)
        print(f"-> Saved yearly aggregated data for {year} to {output_path_year}")
        print(f"  Columns ({len(yearly_aggregated_df.columns)}): {list(yearly_aggregated_df.columns)}\n")



if __name__ == "__main__":
    download_and_parse_parquet()