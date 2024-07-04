from datetime import datetime, timezone, timedelta
import pandas as pd
import os
import math

data_dir = '1_DATA_COLLECTION_SCRIPTS/cryptodata'

#Function for calculating the 1 hour average of the news data
def hour_average(df_timestamps, df_news, value_column,name):
    df_timestamps['timestamp'] = pd.to_datetime(df_timestamps['timestamp'])
    df_news['timestamp'] = pd.to_datetime(df_news['timestamp'])

    hour_values = []

    for index, row in df_timestamps.iterrows():

        current_hour = row['timestamp'] + timedelta(hours=1)
        start_hour = row['timestamp'] 
        filtered_news = df_news[(df_news['timestamp'] >= start_hour) & (df_news['timestamp'] <= current_hour)]

        if not filtered_news.empty:
            moving_average = filtered_news[value_column].mean()
            hour_values.append(moving_average)
        else:
            hour_values.append(None)
    
    df_timestamps[name] = hour_values

    return df_timestamps

#Function for calculating the simple moving average of the news data 
def simple_moving_average(df_timestamps, df_news, value_column, window_size,name):
    df_timestamps['timestamp'] = pd.to_datetime(df_timestamps['timestamp'])
    df_news['timestamp'] = pd.to_datetime(df_news['timestamp'])

    SMA_values = []

    for index, row in df_timestamps.iterrows():

        current_hour = row['timestamp'] + timedelta(hours=1)
        start_hour = current_hour - timedelta(hours=window_size)
        filtered_news = df_news[(df_news['timestamp'] >= start_hour) & (df_news['timestamp'] <= current_hour)]

        if not filtered_news.empty:
            moving_average = filtered_news[value_column].mean()
            SMA_values.append(moving_average)
        else:
            SMA_values.append(None)
    
    df_timestamps[name] = SMA_values

    return df_timestamps

#Function for calculating the exponential moving average of the news data
def exponential_moving_average(df_timestamps, df_news, value_column, window_size, name):
    df_timestamps['timestamp'] = pd.to_datetime(df_timestamps['timestamp'])
    df_news['timestamp'] = pd.to_datetime(df_news['timestamp'])

    EMA_values = []

    for index, row in df_timestamps.iterrows():
        
        current_hour = row['timestamp'] + timedelta(hours=1)
        start_hour = current_hour - timedelta(hours=window_size)
        filtered_news = df_news[(df_news['timestamp'] >= start_hour) & (df_news['timestamp'] <= current_hour)]
        
        smoothing_length = len(filtered_news) 
        alpha = 2 / (smoothing_length + 1)

        if not filtered_news.empty:
            filtered_news = filtered_news.sort_values(by='timestamp') #!flips the order of the data. so the oldest news is at the top (we want this for EMA)
            EMA_previous = filtered_news[value_column].iloc[0]  #!Initialize EMA with the first value in the window
            EMA_current = EMA_previous #!Needed, because when the filtered news = empty then the EMA_current is not defined and we get a error
            for value in filtered_news[value_column].iloc[1:]:
                EMA_current = (value - EMA_previous) * alpha + EMA_previous
                EMA_previous = EMA_current
            EMA_values.append(EMA_current)
        else:
            EMA_values.append(None)
    
    df_timestamps[name] = EMA_values

    return df_timestamps

def process_news_ma_data(start, end, crypto, moving_hours):
    
    # format the crypto and get the news data from csv file
    crypto = [f"{crypto[0]}USD"]
    final_news_df = pd.read_csv(os.path.join(data_dir, f"{crypto[0]}_news.csv"))

    # Create a list to store the hours
    hours = []

    # Calculate the number of hours between start and end
    hours_diff = math.ceil((end - start).total_seconds() / 3600)

    # Iterate over each hour and create a row
    for i in range(hours_diff):
        hour = start + timedelta(hours=i)
        hours.insert(0, {
            "timestamp": hour.isoformat(),
        })

    # Create a DataFrame from the hours list 
    hours_df = pd.DataFrame(hours)

    # Calculate the moving averages
    HA_positive = hour_average(hours_df, final_news_df, 'Positive', 'HA_positive')
    HA_negative = hour_average(hours_df, final_news_df, 'Negative', 'HA_negative')
    HA_neutral = hour_average(hours_df, final_news_df, 'Neutral', 'HA_neutral')

    SMA_positive = simple_moving_average(hours_df, final_news_df, 'Positive', moving_hours, f'SMA_{moving_hours}H_positive')
    SMA_negative = simple_moving_average(hours_df, final_news_df, 'Negative', moving_hours, f'SMA_{moving_hours}H_negative')
    SMA_neutral = simple_moving_average(hours_df, final_news_df, 'Neutral', moving_hours, f'SMA_{moving_hours}H_neutral')

    EMA_positive = exponential_moving_average(hours_df, final_news_df, 'Positive', moving_hours, f'EMA_{moving_hours}H_positive')
    EMA_negative = exponential_moving_average(hours_df, final_news_df, 'Negative', moving_hours, f'EMA_{moving_hours}H_negative')
    EMA_neutral = exponential_moving_average(hours_df, final_news_df, 'Neutral', moving_hours, f'EMA_{moving_hours}H_neutral')

    #save data to csv file 
    file_path = os.path.join(data_dir, f"{crypto[0]}_MA_news.csv")
    hours_df.to_csv(file_path, index=False)
    print(f'Hours for {crypto} saved to {file_path}')
    return hours_df

#! for running and testing the code directly
if __name__ == '__main__':
    crypto = ["BTC"]
    start_time = "2024-07-01 00:00:00"
    end_time = "2024-07-02 00:00:00"   
    moving_hours = 6 

    start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    process_news_ma_data(start, end, crypto, moving_hours)
