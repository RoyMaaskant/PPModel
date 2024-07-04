# for data visualization
import seaborn as sns 
from pylab import rcParams 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# for data manipulation
import numpy as np 
import pandas as pd 
import os

#progress bar
from tqdm import tqdm 

#for the LSTM model
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler

#region setting the style of the visualization
plt.rcParams['figure.dpi'] = 150 
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 10, 6
tqdm.pandas() 
#endregion

#creating sequences of the input data to train validate and test the model
def create_sequences(input_data: pd.DataFrame, target_column, sequence_length):
    sequences = []
    data_size = len(input_data)

    for i in tqdm(range(data_size - sequence_length)):
        sequence = input_data.iloc[i : i + sequence_length] 
        label_position = i + sequence_length
        label = input_data.iloc[label_position][target_column]

        sequences.append((sequence, label))

    return sequences

#creating dataset with torch to handle the sequence data 
class CryptoDataset(Dataset): 
    
    def __init__(self, sequences): #? called constructor
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx] 
        return {
            "sequence": torch.tensor(sequence.to_numpy(), dtype=torch.float32), 
            "label": torch.tensor(label, dtype=torch.float32)  
        }

#Pytorch Lightning data module to manage the data loading using the build in DataLoader
class CryptoPriceDataModule(pl.LightningDataModule):

    def __init__(self, train_sequences, test_sequences, batch_size=8):

        super().__init__()
    
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = CryptoDataset(self.train_sequences)
        self.test_dataset = CryptoDataset(self.test_sequences)
    
    def train_dataloader(self): 
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0) 
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0)

#Setting up and defining the basic parameters for the LSTM model
class PP_Model (nn.Module):

    def __init__(self, n_features, n_hidden=128, n_layers=2):
        super().__init__()

        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers, 
            batch_first=True, 
            dropout=0.2 
        )

        self.regressor = nn.Linear(n_hidden, 1) 

    def forward(self, x): 
        self.lstm.flatten_parameters() 

        lstm_out, (hidden, cell) = self.lstm(x)   
        out = hidden[-1]
        return self.regressor(out)

#Pytorch Lightning model to train the LSTM model
class Crypto_PP_Model(pl.LightningModule):

    def __init__(self, n_features: int):
        super().__init__()

        self.model = PP_Model(n_features)
        self.criterion = nn.SmoothL1Loss() 

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels.unsqueeze(dim=1))
        return loss, output
    
    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss 
    
    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

#descaler to descale the test data predictions and labels
def descale(descaler, values):
    values_2d = np.array(values)[:, np.newaxis] 
    return descaler.inverse_transform(values_2d).flatten()


if __name__ == "__main__":
    #setting the seed for reproducibility
    pl.seed_everything(42)

    #setting the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    #loading the data
    #? This is an inefficient way to load the data, but I want to ensure that the code is robust so that in the future, I can use the same script to predict multiple cryptocurrencies.
    data_dir = '3_THE_PP_MODEL_DATA'
    crypto = ["BTC"]
    interval_length = ['1m', '5m', '15m', '1h']
    data_df = pd.read_csv(os.path.join(data_dir, f"{crypto[0]}-EUR_{interval_length[3]}_data.csv"))
    data_df = data_df.sort_values(by='Timestamp').reset_index(drop=True) 

    #preprocessing the data
    data_df = data_df.drop(columns=['Hourstamp'])
    data_df['Timestamp'] = pd.to_datetime(data_df['Timestamp']) 
    data_df['Timestamp'] = data_df['Timestamp'].apply(lambda x: x.timestamp()) 

    data_df["prev_close"] = data_df["Close"].shift(1) #? shift the close price by 1
    data_df["close_change"] = data_df.progress_apply(lambda row: 0 if np.isnan(row.prev_close) else row.Close - row.prev_close, axis=1) 
    data_df.fillna(0, inplace=True)

    data_df['day_of_week'] = pd.to_datetime(data_df['Timestamp'], unit='s').dt.dayofweek
    data_df['day_of_month'] = pd.to_datetime(data_df['Timestamp'], unit='s').dt.day
    data_df['week_of_year'] = pd.to_datetime(data_df['Timestamp'], unit='s').dt.isocalendar().week
    data_df['month'] = pd.to_datetime(data_df['Timestamp'], unit='s').dt.month

    #set test data
    test_df = data_df
    
    #scaling the data
    set_scaler0 = MinMaxScaler(feature_range=(0, 1))
    scaler0 = set_scaler0.fit(test_df)
    test_df_scaled = scaler0.transform(test_df)
    test_df_scaled = pd.DataFrame(test_df_scaled, columns=test_df.columns)

    #creating sequences from the test data
    SEQUENCE_LENGTH = 300
    test_sequences = create_sequences(test_df_scaled, target_column="Close", sequence_length=SEQUENCE_LENGTH)

    #Load the trained model from the best checkpoint
    best_checkpoint = '3_THE_PP_MODEL_DATA/best-checkpoint-ever.ckpt'
    N_FEATURES = test_df_scaled.shape[1]

    #initialize the model
    trained_model = Crypto_PP_Model.load_from_checkpoint(
        best_checkpoint,
        n_features=N_FEATURES
    )
     
    trained_model.to(device)
    trained_model.freeze() 

    #create the test data module
    test_dataset = CryptoDataset(test_sequences)  

    #generate the predictions using the trained model
    predictions = []
    labels = []

    for item in tqdm(test_dataset):
        sequence = item["sequence"].to(device)
        label = item["label"].to(device)

        output = trained_model.model(sequence.unsqueeze(0)) 
        predictions.append(output.item())
        labels.append(label.item())

    #inverse scaling the predictions and the labels
    descaler = MinMaxScaler()
    descaler.min_, descaler.scale_ = scaler0.min_[4], scaler0.scale_[4] 

    predictions_descaled = descale(descaler, predictions)
    labels_descaled = descale(descaler, labels)

    #prep data for visualization
    test_data = data_df
    test_sequence_data = test_data.iloc[SEQUENCE_LENGTH:]
    test_sequence_data = test_sequence_data.copy()
    test_sequence_data['date'] = pd.to_datetime(test_sequence_data['Timestamp'], unit='s')
    dates = test_sequence_data['date'].apply(lambda x: mdates.date2num(x)).to_list()
    
    # Visualize the predictions
    plt.plot(dates, predictions_descaled, "-", label="Predictions")
    plt.plot(dates, labels_descaled, "-", label="True Values")
    plt.legend()
    plt.show()