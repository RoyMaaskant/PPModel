# for data visualization
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import seaborn as sns 
from pylab import rcParams 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from matplotlib import rc 

# for data manipulation
import math 
import numpy as np 
import pandas as pd 
import os
import time
import glob

#progress bar
from tqdm import tqdm 

#Set environment variable to disable OneDNN optimizations. prevents computations error because off rounding errors
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set float32 matrix multiplication precision to 'medium' or 'high' for better performance on Tensor Cores
import torch
torch.set_float32_matmul_precision('high')

#for the LSTM model
import tensorflow as tf
import torch 
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


#setting the style of the visulaizations
plt.rcParams['figure.dpi'] = 150 #?learned something new:) : sns is statistical data visualization library based on matplotlib, so this line sets the dpi for sns plots.
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 6, 4
tqdm.pandas() #?tqdm is a library that allows you to show a progress bar when you run a loop. It's really useful when you have a long loop and you want to know how long it will take to finish.


#creating sequences to train the model
def create_sequences(input_data: pd.DataFrame, target_column, sequence_length, future_length):
    sequences = []
    data_size = len(input_data)

    for i in tqdm(range(sequence_length, data_size - future_length +1)):
        sequence = input_data.iloc[i - sequence_length: i] #? The iloc method is used to select rows by their integer-location indices
        label = input_data.iloc[i: i + future_length][target_column]

        sequences.append((sequence, label))
        
    return sequences


#build Pytorch dataset and dataloader with the sequences
class CryptoDataset(Dataset): #! explain this class #? overriding the Dataset class from Pytorch
    
    def __init__(self, sequences): #? called constructor
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx] 
        return {
            "sequence": torch.tensor(sequence.to_numpy(), dtype=torch.float32), 
            "label": torch.tensor(label.to_numpy(), dtype=torch.float32) 
        }

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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0) #?workers help to load
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0)

#build the LSTM model
class PP_Model (nn.Module):

    def __init__(self, n_features, n_hidden=128, n_layers=2, future_length=24):
        super().__init__()

        self.n_hidden = n_hidden
        self.future_length = future_length

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers, #? stack multiple LSTM layers on top of each other
            batch_first=True, #? the input and output tensors are provided as (batch, seq, feature)
            dropout=0.2 #? dropout layer to prevent overfitting
        )

        self.regressor = nn.Linear(n_hidden, future_length) #? the output layer

    def forward(self, x): #! get in more detail of the output! here we can optimize more (look into pytorch documentation)
        self.lstm.flatten_parameters()
        lstm_out, (hidden, cell) = self.lstm(x)
        out = lstm_out[:, -1, :]  # Use the last LSTM output for prediction
        out = self.regressor(out)
        return out

class Crypto_PP_Model(pl.LightningModule):

    def __init__(self, n_features: int, future_length: int):
        super().__init__()

        self.model = PP_Model(n_features, future_length=future_length)
        self.criterion = nn.SmoothL1Loss() #huber loss

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    
    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self(sequences, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss #!{"loss" : loss}
    
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
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
    
    # def configure_optimizers(self):
    #     return optim.AdamW(self.parameters(), lr=0.0001)
    

def descale(descaler, values):
    values_2d = np.array(values).reshape(-1, 1)
    return descaler.inverse_transform(values_2d).flatten()


if __name__ == "__main__":

    # Record the start time
    start_time = time.time()

    #setting the seed for reproducibility
    pl.seed_everything(42)


    #setting the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    #region data preprocessing

    #loading the data
    data_dir = 'cryptodata_train'
    crypto = ["BTC"]
    interval_length = ['1m', '5m', '15m', '1h']
    data_df = pd.read_csv(os.path.join(data_dir, f"{crypto[0]}-EUR_{interval_length[3]}_data.csv"))
    data_df = data_df.sort_values(by='Timestamp').reset_index(drop=True) #? sort the data by oldest timestamp on the top(switching the order of the data)


    #preprocessing the data
    data_df = data_df.drop(columns=['Hourstamp'])
    data_df['Timestamp'] = pd.to_datetime(data_df['Timestamp']) 
    data_df['Timestamp'] = data_df['Timestamp'].apply(lambda x: x.timestamp()) #? convert the timestamp to a float (unix timestamp)

    data_df["prev_close"] = data_df["Close"].shift(1) #? shift the close price by 1
    data_df["close_change"] = data_df.progress_apply(lambda row: 0 if np.isnan(row.prev_close) else row.Close - row.prev_close, axis=1) #? calculate the price change
    data_df.fillna(0, inplace=True)

    data_df['day_of_week'] = pd.to_datetime(data_df['Timestamp'], unit='s').dt.dayofweek
    data_df['day_of_month'] = pd.to_datetime(data_df['Timestamp'], unit='s').dt.day
    data_df['week_of_year'] = pd.to_datetime(data_df['Timestamp'], unit='s').dt.isocalendar().week
    data_df['month'] = pd.to_datetime(data_df['Timestamp'], unit='s').dt.month



    #split into train and validation datasets
    train_size = int(len(data_df) * 0.9) #!not needed if we use train_test_split
    train_df, test_df = train_test_split(data_df, test_size=0.1, random_state=42, shuffle=False) 

    set_scaler0 = MinMaxScaler(feature_range=(0, 1))
    scaler0 = set_scaler0.fit(train_df)

    train_df_scaled = scaler0.transform(train_df)
    test_df_scaled = scaler0.transform(test_df)

    train_df_scaled = pd.DataFrame(train_df_scaled, columns=train_df.columns)
    test_df_scaled = pd.DataFrame(test_df_scaled, columns=test_df.columns)


    #? test data so i could test what it does. Before i did not understand why and how the sequence was needed and worked. 
    #? Sequence takes a chunk of the training data and  the label is the next value in the data the data that needs to be predicted 
    sample_data = pd.DataFrame({"feature_1": [1, 2, 3, 4, 5, 6, 7, 8], "feature_2": [6, 7, 8, 9, 10, 11, 12, 13], "target": [11, 12, 13, 14, 15, 16, 17, 18]})
    print(sample_data.head())
    sample_sequence = create_sequences(sample_data, target_column="target", sequence_length=3, future_length=2)
    print(sample_sequence)
    print (sample_sequence[0][0]) #print the first sequence
    print (sample_sequence[0][1]) #print the first label
    print(len(sample_sequence))

    #endregion

    SEQUENCE_LENGTH = 300 #! The longer the sequence the more data the model has to learn from. Can become more accurate but also more computationally expensive or overfitting.
    FUTURE_LENGTH = 24
    train_sequences = create_sequences(train_df_scaled, target_column="Close", sequence_length=SEQUENCE_LENGTH, future_length=FUTURE_LENGTH)
    test_sequences = create_sequences(test_df_scaled, target_column="Close", sequence_length=SEQUENCE_LENGTH, future_length=FUTURE_LENGTH)

    
    N_EPOCHS = 50
    BATCH_SIZE = 400 #? the number of sequences that are passed through the model at once
    N_FEATURES = train_df_scaled.shape[1]

    data_module = CryptoPriceDataModule(train_sequences, test_sequences, batch_size=BATCH_SIZE)
    data_module.setup()

    model = Crypto_PP_Model(n_features=N_FEATURES, future_length=FUTURE_LENGTH)

    #? testing the dataloader
    train_dataset = CryptoDataset(train_sequences)

    for item in data_module.train_dataloader():
        print(item.keys())
        print(item["sequence"].shape)
        print(item["label"].shape) #? output = torch.Size([]), need to deal with this in the model
        # print(item["label"])
        break

    #running the model 
    #? checkpoint_callback: saves the best model based on the validation loss during the training
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="best-checkpoint",
        verbose=True,
        save_top_k=1, #? save only the best model!
        mode="min"
    )

    logger = TensorBoardLogger("lightning_logs", name="crypto_price_prediction")

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)

    #? pytoch lightning trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=N_EPOCHS,
        enable_progress_bar=True
    )

    trainer.fit(model, data_module)

    # Calculate and print the runtime in minutes
    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"Total runtime: {total_time:.2f} minutes")
    print(f"Best Validation Loss: {checkpoint_callback.best_model_score:.4f}")
    

    #! -----------------------------------------------------------------------------------------------------------------------------------------------------
    #Testing the model!

    checkpoint_dir = 'checkpoints/'
    checkpoint_files = glob.glob(checkpoint_dir + '*.ckpt')
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime, default=None)
    print(latest_checkpoint)

    trained_model = Crypto_PP_Model.load_from_checkpoint(
        latest_checkpoint,
        n_features=N_FEATURES,
        future_length=FUTURE_LENGTH
    )
     
    trained_model.to(device)
    trained_model.freeze() #? freeze the model to prevent further training and make predictions faster

    test_dataset = CryptoDataset(test_sequences)  
    # test_dataloader = data_module.test_dataloader()    

    predictions = []
    labels = []

    for item in tqdm(test_dataset):
        sequence = item["sequence"].to(device)
        label = item["label"].to(device)

        output = trained_model.model(sequence.unsqueeze(0)).squeeze().detach().cpu().numpy()
        predictions.append(output)
        labels.append(label.detach().cpu().numpy())

    # Extract the first value of each predicted sequence
    predictions_first = [p[0] for p in predictions]
    labels_first = [l[0] for l in labels]

    # Descale the predictions and labels
    predictions_descaled_first = descale(scaler0, predictions_first)
    labels_descaled_first = descale(scaler0, labels_first)

    print('print(len(predictions)): ', (len(predictions)))
    print('print(len(labels)): ', (len(labels)))
    print('predictions.head()	: ',    predictions[:10]	)
    print('labels.head()	: ',    labels[:10]	)

    #inverse scaling the data
    descaler = MinMaxScaler()
    descaler.min_, descaler.scale_ = scaler0.min_[4], scaler0.scale_[4] #? inverse scaling the close price

    predictions_descaled = descale(descaler, predictions)
    labels_descaled = descale(descaler, labels)

    print(labels_descaled[:10])
    print(predictions_descaled[:10])

    test_data = data_df[train_size:]
    # print(len(test_data))
    # print(len(predictions_descaled))
    # print(len(test_df))

    # test_sequence_data = test_data.iloc[SEQUENCE_LENGTH:]
    # print(test_sequence_data.head())
    # print(len(test_sequence_data))

    # test_sequence_data = test_sequence_data.copy()
    # test_sequence_data['date'] = pd.to_datetime(test_sequence_data['Timestamp'], unit='s')
    # dates = test_sequence_data['date'].apply(lambda x: mdates.date2num(x)).to_list()

    # print(test_sequence_data.head())
    
    # plt.plot_date(dates, predictions_descaled, "-", label="Predictions")
    # plt.plot_date(dates, labels_descaled, "-", label="True Values")
    # plt.legend()
    # plt.show()

  




