import pandas as pd
from datetime import datetime

# Define the Bitcoin halving dates
halving_dates = {
    "date": [
        datetime.strptime("2009-01-03 06:15:05", "%Y-%m-%d %H:%M:%S").replace(tzinfo=None),
        datetime.strptime("2012-11-28 03:24:38", "%Y-%m-%d %H:%M:%S").replace(tzinfo=None),
        datetime.strptime("2016-07-09 04:46:13", "%Y-%m-%d %H:%M:%S").replace(tzinfo=None),
        datetime.strptime("2020-05-11 07:23:43", "%Y-%m-%d %H:%M:%S").replace(tzinfo=None),
        datetime.strptime("2024-04-20 12:09:27", "%Y-%m-%d %H:%M:%S").replace(tzinfo=None)
    ]
}

# Function to find the closest halving and calculate the time difference
def calculate_t_plus_halving(timestamp):
    past_halvings = [halving for halving in halving_dates["date"] if halving< timestamp]
    closest_halving = min(past_halvings, key=lambda x: abs(x - timestamp)) 
    time_difference = timestamp - closest_halving
    time_difference = time_difference.total_seconds() 
    return time_difference


#! below is for testing the function
if __name__ == '__main__':
    
    # Sample DataFrame with random timestamp column
    data = {
        "timestamp": [
        datetime.strptime("2012-11-30 03:24:38", "%Y-%m-%d %H:%M:%S").replace(tzinfo=None),
        datetime.strptime("2016-07-19 04:46:13", "%Y-%m-%d %H:%M:%S").replace(tzinfo=None),
        datetime.strptime("2020-05-10 07:23:43", "%Y-%m-%d %H:%M:%S").replace(tzinfo=None),
        datetime.strptime("2024-04-20 12:09:27", "%Y-%m-%d %H:%M:%S").replace(tzinfo=None)
        ]
    }
    df = pd.DataFrame(data)

    # Add a new column with the time difference to the closest halving
    df["t_plus_halving"] = df["timestamp"].apply(calculate_t_plus_halving)
    print(df)
