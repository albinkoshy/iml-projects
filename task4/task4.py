# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RationalQuadratic, Matern, RBF

from sklearn.pipeline import make_pipeline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None
    
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("public/pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("public/pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("public/train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("public/train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("public/test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraing data 
        # and then used to extract features from the training and test data.
        self.encoder = nn.Linear(1000, 512)
        self.decoder = nn.Linear(512, 1000)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture 
        # defined in the constructor.
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def make_feature_extractor(x, y, batch_size=256, eval_size=1000):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # Pretraining data loading
    in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)
    
    # model declaration
    model = Net()
    model.to(device)
    model.train()
    
    # TODO: Implement the training loop. The model should be trained on the pretraining data. Use validation set 
    # to monitor the loss.
    
    dataset = TensorDataset(x_tr.type(torch.float), y_tr.type(torch.float))
    
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size, 
                              shuffle=True, 
                              pin_memory=True, num_workers=2)
    
    # Train the autoencoder
    print("Training Autoencoder...")
    
    n_epochs = 50
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        
        train_loss = 0.0
        
        # Training Loop
        model.train()
        for (x, y) in train_loader:
            x = x.to(device)           
            output = model(x)
            loss = loss_function(output, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
        
        train_loss /= len(dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            x_val = x_val.to(device)
            output = model(x_val)
            val_loss = loss_function(output, x_val)
            
        val_losses.append(val_loss.item())
        
        print(f"Epoch: {epoch+1}\t Training loss: {train_loss}\t Validation loss: {val_loss}")
    
    # Remove the decoder part of the autoencoder model and add layers to train with y_pretrain
    new_model = nn.Sequential(*list(model.children())[:-1])
    new_model.add_module('new1', nn.Linear(512, 256))
    new_model.add_module('new2', nn.ReLU())
    new_model.add_module('new3', nn.Linear(256, 1))
    new_model.to(device)
                
    # Train this new model
    print("Training model with labels y_pretrain...")

    n_epochs = 35
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(new_model.parameters(), lr=1e-4)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        
        train_loss = 0.0
        
        # Training Loop
        new_model.train()
        for (x, y) in train_loader:
            x, y = x.to(device), y.to(device)         
            output = new_model(x)
            loss = loss_function(output, y.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
        
        train_loss /= len(dataset)
        train_losses.append(train_loss)
        
        # Validation
        new_model.eval()
        with torch.no_grad():
            x_val, y_val = x_val.to(device), y_val.to(device)
            output = new_model(x_val)
            val_loss = loss_function(output, y_val.unsqueeze(1))
            
        val_losses.append(val_loss.item())
        
        print(f"Epoch: {epoch+1}\t Training loss: {train_loss}\t Validation loss: {val_loss}")
    
    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        # TODO: Implement the feature extraction, a part of a pretrained model used later in the pipeline.
        x = torch.tensor(x, dtype=torch.float)
        final_model = nn.Sequential(*list(new_model.children())[:-2])
        final_model.to(device)
        
        final_model.eval()
        with torch.no_grad():
            x = x.to(device)
            x_features = final_model(x)
            x_features = x_features.cpu().detach().numpy()
        return x_features

    return make_features

def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline
    
    input: feature_extractors: dict, a dictionary of feature extractors

    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X):
            assert self.feature_extractor is not None
            X_new = feature_extractors[self.feature_extractor](X)
            return X_new
        
    return PretrainedFeatures

def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.
    kernel = RationalQuadratic() * Matern()
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    # Utilize pretraining data by creating feature extractor which extracts lumo energy 
    # features from available initial features
    feature_extractor =  make_feature_extractor(x_pretrain, y_pretrain)
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    
    # regression model
    regression_model = get_regression_model()

    y_pred = np.zeros(x_test.shape[0])
    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.

    pipeline = make_pipeline(PretrainedFeatureClass(feature_extractor="pretrain", mode=None), regression_model)
    
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test.to_numpy())
    
    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")