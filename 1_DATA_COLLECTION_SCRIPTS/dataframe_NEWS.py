from datetime import datetime, timezone
import requests
import pandas as pd
import os
import torch
from termcolor import colored
from finbert_NEWS import SentimentAnalyzer


data_dir = '1_DATA_COLLECTION_SCRIPTS/cryptodata'

#The Alpaca API is used to retrieve news data and format it into a pandas dataframe
class AlpacaImplementation:

    url = "https://data.alpaca.markets/v1beta1/news"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": "",
        "APCA-API-SECRET-KEY": ""
        }

    def get_news(self, symbol, start, end, page_token = None): #! page_token is used to get the next page of news. First set to None so we retrieve the first page 
        start = start.isoformat()
        end = end.isoformat()
        params={
            "symbols": symbol, 
            "start": start, 
            "end": end, 
            "exclude_contentless": True, 
            "limit": 50, 
            "include_content": False
            }
        if page_token: #!if true, page_token is added to the params. this only happens when there are more pages than 1 (so more then 50 news items)
            params["page_token"] = page_token 

        #error handling for the request
        response = requests.get(
            self.url, 
            headers=self.headers, 
            params=params
             )
        
        print(f"Response Status Code: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: {response.json()}")
        return response.json()

    #news to pandas dataframe
    def format_news_to_dataframe(self, news_json):
        if "news" in news_json:
            news_list = news_json["news"] #! Get the list of news
            formatted_news = []
            for news in news_list:
                summary = news.get("summary", "") 
                if not summary: #! Replace None or empty values with an empty string and remove newlines
                    summary = ""
                summary = summary.replace("\n", " ").replace("\r", " ") #!add space to ensure NO new lines are made. 
                formatted_news.append({
                    "timestamp": news.get("created_at", ""),
                    "symbol": news.get("symbols", ""), 
                    "headline": news.get("headline", ""),
                    "summary": summary,
                    "source": news.get("source", ""),
                    "url": news.get("url", "")
                })
            df = pd.DataFrame(formatted_news)
            return df
        else:
            return pd.DataFrame([])
        
    def getRemainingLimit(self):
        #?Many APIs have a rate limit. I wanted to check if repeatedly requesting with the next page_token (if there were more than 50 news articles) would affect the rate limit. It did not.
        response = requests.get(
            self.url, 
            headers=self.headers, 
        )
        
        ratelimit_remaining = int(response.headers['X-RATELIMIT-REMAINING'])
        ratelimit_reset = int(response.headers['X-RATELIMIT-RESET'])
        
        print(colored(f'X-Ratelimit-Remaining = {ratelimit_remaining}', 'yellow'))
        print(colored(f'X-Ratelimit-Reset     = {ratelimit_reset}', 'yellow'))
      
#retrieve, process and save the news data
def process_news_data(start, end, crypto):

    # format the crypto so it can be used in the Alpaca API
    crypto = [f"{crypto[0]}USD"]

    #initialize the classes
    Alpaca = AlpacaImplementation()
    Analyse = SentimentAnalyzer()

    #get the news data
    all_news_data = []
    page_token = None

    while True:
        news_json = Alpaca.get_news(symbol=crypto[0], start=start, end=end, page_token=page_token)
        news_df = Alpaca.format_news_to_dataframe(news_json)
        Alpaca.getRemainingLimit()

        #print("Formatted DataFrame:\n", news_df)  #! Debug print statement
        
        all_news_data.append(news_df)

        page_token = news_json.get("next_page_token")
        if not page_token:
            break
    
    #creates the columns and performs sentiment analysis on the news data
    if all_news_data:
        final_news_df = pd.concat(all_news_data, ignore_index=True) #! Concatenate all the dataframes into one
    
        # Initialize columns for sentiment results
        final_news_df['Positive'] = None
        final_news_df['Negative'] = None
        final_news_df['Neutral'] = None
        final_news_df['Max_Prob'] = None
        final_news_df['Sentiment'] = None

        # Perform sentiment analysis on each headline and summary
        print("GPU avaliable:", torch.cuda.is_available(),"\n") #! check if cuda (GPU) is available for sentiment analysis

        for index, row in final_news_df.iterrows():
            headline = row['headline']
            summary = row['summary']
            sentiment_result = Analyse.estimate_sentiment([headline, summary])
            final_news_df.at[index, 'Positive'] = sentiment_result['Positive'][0]
            final_news_df.at[index, 'Negative'] = sentiment_result['Negative'][0]
            final_news_df.at[index, 'Neutral'] = sentiment_result['Neutral'][0]
            final_news_df.at[index, 'Max_Prob'] = sentiment_result['Max_Prob'][0]
            final_news_df.at[index, 'Sentiment'] = sentiment_result['Sentiment'][0]

        #save data to csv file 
        file_path = os.path.join(data_dir, f"{crypto[0]}_news.csv") 
        final_news_df.to_csv(file_path, index=False) 
        print(f'News for {crypto}  saved to {file_path}')

    return final_news_df
    
#! for running and testing the code directly
if __name__ == '__main__':
    crypto = ["BTC"]
    start_time = "2024-07-01 00:00:00"
    end_time = "2024-07-02 00:00:00"   

    start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    
    process_news_data(start, end, crypto) 

   