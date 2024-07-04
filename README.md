The Crypto PP Model
=====================
In this project, I aim to build a Price Prediction Model (PPModel) in Python to forecast the prices of various cryptocurrencies using a Long Short-Term Memory (LSTM) model. The model integrates a sentiment analysis on the collected news related to a specific cryptocurrency to determine the impact of news trends and hype on the coins price. Additionally, the Bitcoin halving’s are a significant event in the cryptocurrency world and often have a substantial impact on news and market hype. Based on this information, I am building an LSTM model to predict the future closing prices.  
<br>
This project is for the course ‘Programming Skills Development for IEM’  
<br>


Table of Content
------------------
-   Note!
-   Get started
-   The project
-   Features of the Crypto PP Model
-   Usage
<br>


Note!
-------
Before we start, I would like to address some points: 

-	To make the scope of the project more manageable, we will focus only on data from Bitcoin. During the data collection and training phase, this will save time and resources.
<br>

-	During the final phase of this project, I realized that I had taken a fundamentally incorrect approach. While testing and using the model, I discovered that I had constructed the sequences incorrectly during training, allowing predictions only one hour ahead. To ensure that the model predicts not just the present but also the future, I am working on a new version of `CRYPTO_LSTM_train.py`.   
To make accurate future predictions, the create_sequence function must generate a list of labels that predict several intervals into the future. Once the code for the LSTM training, validation, and test model is adjusted to handle an array of labels, I will retrain the model and further refine the parameters. The goal is to predict the price up to 24 hours into the future.
<br>

-	In addition, there are several improvements I would like to apply to the model:  
    1.	Training the model on more data from different cryptocurrencies to enhance overall robustness and prevent overfitting caused by training on Bitcoin data only.
    2.	Implementing L1/L2 regularization to mitigate overfitting.
    3.	Adding various indicators and moving averages to the prediction plot based on the news.
    4.	Developing a front-end on a local server to enhance JavaScript skills and improve the usability of the model.
<br>


Get started
-------------------

Step-by-step instructions on how to download the project files and the required dependencies.

1)	First step is to clone the repository to your own environment:  
`git clone https://github.com/RoyMaaskant/PPModel.git` 
<br>

2)  Navigate to the project folder  
`cd PPModel`
<br>

3)	Create and activate virtual environment to install the dependencies:  
    -   Create venv: `python -m venv venv`
    -   Activate venv for Windows: `venv\Scripts\activate`
    -   Activate venv for MacOS or Linux: `source venv/bin/activate`
<br>

4)	Install the required dependencies to your venv:  
`pip install -r requirements.txt`
<br>

5)	Run the files:
    -   Command line warriors: `python <filename>`
    -   IDE people, just press start...
<br>

The project 
----
The primary objective of this project is to develop a Price Prediction Model (PPModel) in Python to forecast the prices of bitcoin using a Long Short-Term Memory (LSTM) model. For me, the project also improves my knowledge about programming in several areas:  
<br>
1.	General (Python) programming skills: Gained a good understanding of the basics and how to use these skills as “building blocks”   to solve new problems.     
2.	Data gathering: learned how to retrieve data using API’s    
3.	Data Processing and Cleaning: Learned how to handle, preprocess and store large datasets.    
4.	Machine learning and recurrent neural networks: learned how to train, validate and test different types of RNNs (mostly LSTM’s)   
<br>

Before starting the Crypto PP Model, I studied the concepts and mathematics of recurrent neural networks to become well known in the relevant terms and concepts, thereby making the larger project more manageable. Additionally, I undertook several smaller practice projects to integrate the mathematical theories with practical implementations using Python, PyTorch, and PyTorch Lightning.    

I followed the Stat Quest course by Joshua Starmer on (recurrent) Neural Networks. A big plus of this course was its excellent visualization of the models and the math behind it, which greatly enhanced my understanding of the topics. This knowledge was also mega useful when troubleshooting coding issues, as understanding the terminology made it much easier for me to identify and solve problems. These are some of the important concepts:   
<br>
-	Probability and error functions: calculating using the gradient descent  
-	Stochastic Gradient Descent (SGD): Optimization for large datasets.  
-	The chain rule and gradient descent for backpropagation (for the optimization of weights and biases)  
-	Activation functions: ReLu, Softmax, ArgMax and Cross-Entropy  
-	Convolutional Neural Networks (CNN) for image classification   
-	Recurrent Neural Networks (RNN) for sequential data  
-	Long Short-Term Memory (LSTM) for handling long-term dependencies in sequences.   
-	Tensors for efficiently storing large amount of data that can be used by (PyTorch) models.  
<br>

Features of the Crypto PP Model  
----
The project consists of three parts: data collection, model training, and using the trained model to predict the current price. The repository includes:       
<br>

**0_raw_files**   
These are the unrefined Python scripts used for testing different elements and functions of the LSTM model. `CRYPTO_LSTM_train(future prediction).py` contains the new and improved version that handles larger labels of the training sequences, enabling us to predict the Bitcoin price further into the future (work in progress).   
<br>

**1_DATA_COLLECTION_SCRIPTS**    
Used for the data collection of cryptocurrencies and the news.     

**`Main.py`**   
Is the main script. You can set all the parameters and calls all the other function in the right order. The main data output that can be used for the training of the LSTM will be a CSV file stored in `1_DATA_COLLECTION_SCRIPTS\cryptodata` named `<crypto>-EUR_<interval>_data`   

**`dataframe_news.py`**    
Retrieves and processes news data for a specific cryptocurrency, calls the finbert sentiment analyses and combines the results in a pandas data frame. The output will be a CSV file stored in `1_DATA_COLLECTION_SCRIPTS\cryptodata` named `<crypto>USD_news`. We use the Alpaca API[1] to fetch the news data. 
[1]: https://docs.alpaca.markets/reference/news-3     

**`finbert_NEWS.py`**    
Performs sentiment analysis on the crypto news using the Finbert model[2]. The function in the script tokenizes the headline and the summary of each article, runs the model, uses softmax to determine the probability that a news article is positive, negative or neutral and returns a data frame with the sentiment scores. 
[2]: https://huggingface.co/ProsusAI/finbert    

**`dataframe_News_MA.py`**    
Builds a new data frame based on the hours in the given interval, calculates for each hourly average, simple moving average (SMA), and exponential moving average (EMA). The output will be a CSV file stored in `1_DATA_COLLECTION_SCRIPTS\cryptodata` named `<crypto>USD _MA_news`.      

**`dataframe_CRYPTO.py`**  
This script retrieves historical cryptocurrency data from the Bitvavo API[3], calculates moving averages, and appends the time difference from the nearest past Bitcoin halving event. The output will be a CSV file stored in `1_DATA_COLLECTION_SCRIPTS\cryptodata` named `<crypto>-EUR_<interval>_data`.   
[3]: https://docs.bitvavo.com/    

**`dataframe_HALVING.py`**   
This script calculates the time difference between timestamps and the closest Bitcoin halving event, then appends this information to a DataFrame.    
<br>
**2_TRAIN_LSTM_MODEL**    
Contains folders where the news checkpoints and the lightning_logs of the model will be saved. The folder `cryptodata_train` is the folder used by the script to get the data to train. Here we can place the named `<crypto>-EUR_<interval>_data` created in step 1.   

**`CRYPTO_LSTM_train.py`**   
This script builds, trains, and evaluates an LSTM model to predict cryptocurrency prices using historical data and sentiment analysis features. It utilizes PyTorch Lightning for model training and includes data preprocessing, sequence creation, model definition, and visualization of predictions.   
<br>

**3_THE_PP_MODEL_DATA**  
Contains the checkpoint of the best trained model and the crypto data that will be used by `THE_PP_MODEL.py` for predicting the future cryptocurrency price.    

Usage
----
**Data collection**   
Use ` 1_DATA_COLLECTION_SCRIPTS\Main.py`. and fill in the given parameters like so:   
```
start_time = "2024-01-01 00:00:00" 
end_time = "2024-07-02 13:00:00" 
moving_hours = 72 
interval_length = ['1h']
crypto = ["BTC"]
```
Run the code. The output will be a CSV file stored in `1_DATA_COLLECTION_SCRIPTS\cryptodata` named `<crypto>-EUR_<interval>_data`   
<br>

**Training a new mode**     
Store the created csv file from the `main.py`and in the folder `cryptodata_train`. Run the script and evaluate the result. Keep tweaking the following parameters to reduce the validation_loss:      
```
n_hidden=128
n_layers=2
train_size = int(len(data_df) * 0.9)
test_size=0.1
lr=0.001
N_EPOCHS = 50
BATCH_SIZE = 400
SEQUENCE_LENGTH = 300
```
<br >

**Predicting the price**   
Save the best checkpoint and `<crypto>-EUR_<interval>_data` in the folder `3_THE_PP_MODEL_DATA`and run the `THE_PP_MODEL.py`to plot the graph with the predictions.    
