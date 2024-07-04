# for data visualization
import seaborn as sns 
from pylab import rcParams 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# for data manipulation
import numpy as np 
import pandas as pd 
import os
import time
import glob

#progress bar
from tqdm import tqdm #! for showing progress bar

#Set environment variable to disable OneDNN optimizations. prevents computations error because off rounding errors
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set float32 matrix multiplication precision to 'medium' or 'high' for better performance on Tensor Cores
import torch
torch.set_float32_matmul_precision('high')

#for the LSTM model
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

#region setting the style of the visualization
plt.rcParams['figure.dpi'] = 150 #! Learned something new:) : sns is statistical data visualization library based on matplotlib, so this line sets the dpi for sns plots.
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
        sequence = input_data.iloc[i : i + sequence_length] #! The iloc method is used to select rows by their integer-location indices
        label_position = i + sequence_length 
        label = input_data.iloc[label_position][target_column]

        sequences.append((sequence, label))
        
    return sequences

#creating dataset with torch to handle the sequence data 
class CryptoDataset(Dataset): #! Overriding the Dataset class from Pytorch
    
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0) #?workers help to load
    
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
            num_layers=n_layers, #! stack multiple LSTM layers on top of each other
            batch_first=True, #! the input and output tensors are provided as (batch, seq, feature)
            dropout=0.2 #! dropout layer to prevent overfitting
        )

        self.regressor = nn.Linear(n_hidden, 1) #! the output layer

    def forward(self, x): #! get in more detail of the output! here we can optimize more (look into pytorch documentation)
        self.lstm.flatten_parameters() #! for more efficient memory usage and multi GPU training

        lstm_out, (hidden, cell) = self.lstm(x)   
        out = hidden[-1]
        return self.regressor(out)

#Pytorch Lightning model to train the LSTM model
class Crypto_PP_Model(pl.LightningModule):

    def __init__(self, n_features: int):
        super().__init__()

        self.model = PP_Model(n_features)
        self.criterion = nn.SmoothL1Loss() #! Huber loss (less sensitive to outliers than the Mean Squared Error loss)

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
        #? Not sure if ReduceLROnPlateau gives different results than a fixed learning rate
        optimizer = optim.AdamW(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
    
    
def descale(descaler, values):
    values_2d = np.array(values)[:, np.newaxis] #! i know my values are 1D, so i add a new axis to make it 2D
    return descaler.inverse_transform(values_2d).flatten()


if __name__ == "__main__":
    # Record the start time
    start_time = time.time()

    #setting the seed for reproducibility
    pl.seed_everything(42)

    #setting the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    #loading the data
    #? This is an inefficient way to load the data, but I want to ensure that the code is robust so that in the future, I can use the same script to predict multiple cryptocurrencies.
    data_dir = '2_TRAIN_LSTM_MODEL/cryptodata_train'
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
    train_size = int(len(data_df) * 0.9) #!use this part add the end for testing and plotting the predictions. Use train_test_split function
    train_df, test_df = train_test_split(data_df, test_size=0.1, random_state=42, shuffle=False) 

    print(f"Training data shape: {train_df.shape}")
    print(f"Validation data shape: {test_df.shape}")
    # print(train_df.head())

    #scaling the data
    #? We attempted to scale different columns based on what best suited their data types. 
    #? This approach became extremely complex and did not yield the desired results, partly due to the difficulty of reversing the scaling. 
    #? We also tested MinMax(-1,1) and one-hot encoding, but all of these methods performed worse than a simple scale from 0 to 1.
    
    set_scaler0 = MinMaxScaler(feature_range=(0, 1))
    scaler0 = set_scaler0.fit(train_df)

    train_df_scaled = scaler0.transform(train_df)
    test_df_scaled = scaler0.transform(test_df)

    train_df_scaled = pd.DataFrame(train_df_scaled, columns=train_df.columns)
    test_df_scaled = pd.DataFrame(test_df_scaled, columns=test_df.columns)

    #creating sequences from the descaled data
    SEQUENCE_LENGTH = 300 #! The longer the sequence the more data the model has to learn from. Can become more accurate but also more computationally expensive or sensitive for overfitting.
    train_sequences = create_sequences(train_df_scaled, target_column="Close", sequence_length=SEQUENCE_LENGTH)
    test_sequences = create_sequences(test_df_scaled, target_column="Close", sequence_length=SEQUENCE_LENGTH)
 
    print(f"Training Sequences: {len(train_sequences)}")
    print(f"Validation Sequences: {len(test_sequences)}")

    #setting the hyperparameters
    N_EPOCHS = 50
    BATCH_SIZE = 400 #! the number of sequences that are passed through the model at once
    N_FEATURES = train_df_scaled.shape[1]
    
    #initializing the data module
    data_module = CryptoPriceDataModule(train_sequences, test_sequences, batch_size=BATCH_SIZE)
    data_module.setup()
    model = Crypto_PP_Model(n_features=N_FEATURES)

    #running the model 
    #! checkpoint_callback: saves the best model based on the validation loss during the training
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="2_TRAIN_LSTM_MODEL/checkpoints",
        filename="best-checkpoint",
        verbose=True,
        save_top_k=1, #?!save only the best model!
        mode="min"
    )

    logger = TensorBoardLogger("2_TRAIN_LSTM_MODEL/lightning_logs", name="crypto_price_prediction")

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)

    #Setting up the trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=N_EPOCHS,
        enable_progress_bar=True
    )

    #training the model
    trainer.fit(model, data_module)
    print(f"Best Validation Loss: {checkpoint_callback.best_model_score:.4f}")

    # Calculate and print the runtime in minutes
    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f"Total runtime: {total_time:.2f} minutes")
    
    #Testing the model! -----------------------------------------------------------------------------------------------------------------------------------------------------
    
    #Load the trained model from the most recent checkpoint (the one from this run)
    checkpoint_dir = '2_TRAIN_LSTM_MODEL/checkpoints'
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime, default=None)
    print(latest_checkpoint)

    #initialize the model
    trained_model = Crypto_PP_Model.load_from_checkpoint(
        latest_checkpoint,
        n_features=N_FEATURES
    )
     
    trained_model.to(device)
    trained_model.freeze() #? freeze the model to prevent further training and make predictions faster

    #create the test data module
    test_dataset = CryptoDataset(test_sequences)  

    #generate the predictions using the trained model
    predictions = []
    labels = []

    for item in tqdm(test_dataset):
        sequence = item["sequence"].to(device) #? move the data to the device (GPU)
        label = item["label"].to(device)

        output = trained_model.model(sequence.unsqueeze(0)) #? unsqueeze to add a batch dimension
        predictions.append(output.item())
        labels.append(label.item())

    #inverse scaling the predictions and the labels
    descaler = MinMaxScaler()
    descaler.min_, descaler.scale_ = scaler0.min_[4], scaler0.scale_[4] #? inverse scaling the close price

    predictions_descaled = descale(descaler, predictions)
    labels_descaled = descale(descaler, labels)

    #prep data for visualization
    test_data = data_df[train_size:]
    test_sequence_data = test_data.iloc[SEQUENCE_LENGTH:]
    test_sequence_data = test_sequence_data.copy()
    test_sequence_data['date'] = pd.to_datetime(test_sequence_data['Timestamp'], unit='s')
    dates = test_sequence_data['date'].apply(lambda x: mdates.date2num(x)).to_list()
    
    # Visualize the predictions
    plt.plot(dates, predictions_descaled, "-", label="Predictions")
    plt.plot(dates, labels_descaled, "-", label="True Values")
    plt.legend()
    plt.show()




