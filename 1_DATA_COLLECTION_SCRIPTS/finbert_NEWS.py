from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List
import pandas as pd

#performance sentiment analysis on the retrieved news data. Get called by process_news_data function
#? We use a pretrained model called Finbert. Finbert is a pre-trained NLP model that is fine-tuned on financial data.
class SentimentAnalyzer:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)
        self.labels = ["positive", "negative", "neutral"]

    def estimate_sentiment(self, news: List[str]) -> pd.DataFrame:
        if news:
            tokens = self.tokenizer(news, return_tensors="pt", padding=True).to(self.device) #!tokenize the news
            result = self.model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"] #!get the logits from the model
            result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1) #!apply SoftMax to get the probabilities
           
            max_prob = result[torch.argmax(result)]
            sentiment = self.labels[torch.argmax(result)]

            result_list = result.tolist()
            data = {
                "Positive": [result_list[0]],
                "Negative": [result_list[1]],
                "Neutral": [result_list[2]],
                "Max_Prob": [max_prob.item()],
                "Sentiment": [sentiment]
            }

            return pd.DataFrame(data)
        else:
            data = {
                "Positive": [0.0],
                "Negative": [0.0],
                "Neutral": [0.0],
                "Max_Prob": [0.0],
                "Sentiment": [self.labels[-1]]
            }
            return pd.DataFrame(data)

#! for running and testing the code directly
if __name__ == "__main__":
    print(torch.cuda.is_available()) #! model can be run on GPU
    
    #initialize the class
    Analyse = SentimentAnalyzer()

    #perform sentiment analysis on test text
    sentiment_result = Analyse.estimate_sentiment(["Telegram-Associated Notcoin, Sam Altman-Founded Worldcoin Outshine Bitcoin, Ethereum On A Weak Day","Telegram-linked cryptocurrency Notcoin (CRYPTO: NOT) and Sam Altman-founded Wordlcoin (CRYPTO: WLD) raked in major gains on a day when market heavyweights slipped lower."])
    print(sentiment_result)
    