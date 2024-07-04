from python_bitvavo_api.bitvavo import Bitvavo
from datetime import datetime, timezone
import time

import pandas as pd
import os
import math
from termcolor import colored
from dataframe_HALVING import calculate_t_plus_halving

data_dir = '1_DATA_COLLECTION_SCRIPTS/cryptodata'

#Bivavo API implementation to retrieve candle data
#? There are many methods/API's/databases available to retrieve historical data. Bitvavo is used because it is the broker I use for trading.
class BitvavoImplementation:

    def __init__(self):
        self.bitvavo = Bitvavo() 
    
    def get_candles(self, symbol, interval, limit, start, end):
        try: 
            candles = self.bitvavo.candles(symbol=symbol, interval=interval, limit=limit, start=start, end=end)
            # print(candles) #! if no data is retrieved show the error message
            print(f"Retrieved {len(candles)} data points, with intervals of {interval} for {symbol}.", '\n')

            formatted_data = [
                {'Timestamp': candle[0], 'Open': candle[1], 'High': candle[2], 
                 'Low': candle[3], 'Close': candle[4], 'Volume': candle[5]}
                for candle in candles
            ]

            df = pd.DataFrame(formatted_data)  #! Convert data to a pandas DataFrame
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms') #!Convert the timestamp to a datetime object
            return df
        
        #error handling for the request
        except Exception as e:
            print(f"No data retrieved for {symbol}.")
            print(f"An error occurred ! ! : {e}")
            return pd.DataFrame()
        
    #calculate the number of blocks needed to get all the data in blocks of 1440 data points for each time interval
    #? get_candles has a limit of 1440 data points and no page_token attribute to get the next page of data. Therefore we calculate how many blocks of 1440 data points we have in the given time interval
    def calculate_blocks(self, total_minutes, interval, limit):
        if interval.endswith('m'):
            interval_minutes = int(interval[:-1])
        elif interval.endswith('h'):
            interval_minutes = int(interval[:-1]) * 60
        total_intervals = total_minutes // interval_minutes
        blocks = math.ceil(total_intervals / limit)
        return blocks
    
    #get the remaining request limit of the API. 
    def getRemainingLimit(self):
        print(colored(f"Remaining limit: {self.bitvavo.getRemainingLimit()} \n", 'blue'))
        return self.bitvavo.getRemainingLimit()
    
def process_crypto_data(start, end, crypto, interval_length):

    interval_length = interval_length
    limit = 1440  #! Number of data points to retrieve
    iterations = 1  #!Initial value for the number of iterations

    #create an instance of the BitvavoImplementation class
    bitvavo = BitvavoImplementation()

    #create the data directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)

    #Determine the number of iterations we need to get all the data in blocks of 1440 data point for each time interval
    total_minutes = int((end - start).total_seconds() / 60)
    blocks_needed = {interval: bitvavo.calculate_blocks(total_minutes, interval, limit) for interval in interval_length}
    print(blocks_needed)

    # format the crypto for the Bitvavo API and retrieving the csv news file 
    coin =  f"{crypto[0]}-EUR"
    crypto2 = [f"{crypto[0]}USD"]

    #retrieve the candle data for the different time intervals
    for length in interval_length:
        interval = length

        current_end = end
        all_data = pd.DataFrame()

        iterations = blocks_needed.get(length, None)
        
        for _ in range(iterations):
            historical_data = bitvavo.get_candles(symbol=coin, interval=length, limit=limit, start=start, end=current_end)
            bitvavo.getRemainingLimit() #! get remaining limit API (calls per minute)
            
            if historical_data.empty:
                print(f"No data returned for {coin} {interval}.")
                break
            
            all_data = pd.concat([all_data, historical_data])
            current_end = historical_data['Timestamp'].min() - pd.Timedelta(minutes=int(interval[:-1]))
            current_end = current_end.tz_localize('UTC')  #! Ensure the datetime is timezone-aware
            
            print(f'Fetched data up to {current_end} for {coin} with interval {interval}')
            
            if current_end <= start:
                break

        if all_data.empty:
            print(f"No data was collected for {coin} {interval}")
            continue

    
        #add hours to df so we can match the SMA and EMA data. Change the order of the columns because it looks nicer :)
        all_data['Hourstamp'] = pd.to_datetime(all_data['Timestamp']).dt.floor('h')
        all_data['Hourstamp'] = all_data['Hourstamp'].dt.tz_localize('UTC')
        all_data = all_data[['Timestamp', 'Hourstamp', 'Open', 'High', 'Low', 'Close', 'Volume']]       

        #Adding t+ for bitcoin halving
        all_data["t_plus_halving"] = all_data["Timestamp"].apply(calculate_t_plus_halving)

        #add the moving averages to the data
        MA_NEWS_df = pd.read_csv(os.path.join(data_dir, f"{crypto2[0]}_MA_news.csv"))
        MA_NEWS_df['timestamp'] = pd.to_datetime(MA_NEWS_df['timestamp'])
        merged_data = pd.merge(all_data, MA_NEWS_df, left_on='Hourstamp', right_on='timestamp', how='inner')
        merged_data = merged_data.drop(columns=['timestamp'])
    
        #handling the NaN values
        #? tried different methods during the training of the LSTM model. The best results were achieved by filling the NaN values with 0.
        merged_data = merged_data.fillna(0)
        # merged_data = merged_data.ffill().bfill()
        # merged_data = merged_data.infer_objects(copy=False)

        #write to csv file
        file_path = os.path.join(data_dir, f"{coin}_{interval}_data.csv")
        merged_data.to_csv(file_path, index=False)
        print(f'Data for {coin} and interval {interval} saved to {file_path}')

    print(start, end)

    return merged_data

#! for running and testing the code directly
if __name__ == '__main__':
    crypto = ["BTC"]

    start_time = "2024-07-01 00:00:00"
    end_time = "2024-07-02 00:00:00" 

    interval_length = ['1h']

    start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    process_crypto_data(start, end, crypto,interval_length)
