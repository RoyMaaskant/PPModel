from datetime import datetime, timezone
from dataframe_NEWS import process_news_data
from dataframe_NEWS_MA import process_news_ma_data
from dataframe_CRYPTO import process_crypto_data

# Define the parameters for the news and crypto data
start_time = "2024-7-01 00:00:00" 
end_time = "2024-07-02 13:00:00" 
moving_hours = 72 #! Number of hours to calculate the simple and exponential moving averages (SMA and EMA)
interval_length = ['1h']
crypto = ["BTC"] 

# interval_length = ['1m', '5m', '15m', '1h'] #! Initially, I always retrieved data for most types of intervals. Later, during the training of the LSTM, I discovered that the 1-hour timeframe worked best.
# crypto = ["BTC","ETH","SOL"]  #!possible to retrieve data for multiple coins
# crypto = ["JASMY"] #! This is a test coin that does not exist add Bitvavo, use to test error handling

#convert the string to datetime
start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


if __name__ == '__main__': 
    for coin in crypto:
        print(f"\nRunning the code for {coin}\n")
        crypto = [coin]

        #run the different data collection and processing functions

        #! retrieve news data of given coin and executes the sentiment analysis function on the news data
        process_news_data(start, end, crypto) 
        
        #! based on the news data, calculates the different (moving) averages
        process_news_ma_data(start, end, crypto, moving_hours) 
        
        #! retrieves the crypto data of given coin and interval length and appends the moving averages and the bitcoin halving t + time
        process_crypto_data(start, end, crypto, interval_length) 


