# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, ExpSineSquared, ConstantKernel

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train   
    X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df)

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
    
    # Use mean values for NaN
    # train_filled_df = train_df.fillna(train_df.mean())
    # test_filled_df = test_df.fillna(test_df.mean())
    
    # Use interpolated values for NaN
    train_filled_df = train_df.interpolate(limit_direction="both")
    test_filled_df = test_df.interpolate(limit_direction="both")

    # Preprocessing: Encode season attributes using OneHotEncoder
    encoder_train = OneHotEncoder(handle_unknown='ignore')
    encoder_test = OneHotEncoder(handle_unknown='ignore')

    encoder_train_df = pd.DataFrame(encoder_train.fit_transform(train_filled_df[['season']]).toarray())
    encoder_test_df = pd.DataFrame(encoder_test.fit_transform(test_filled_df[['season']]).toarray())

    train_filled_df.drop(['season'], axis=1, inplace=True)
    test_filled_df.drop(['season'], axis=1, inplace=True)

    encoder_train_df.columns = encoder_train.get_feature_names_out()
    encoder_test_df.columns = encoder_test.get_feature_names_out()

    train_final_df = pd.concat([encoder_train_df, train_filled_df], axis=1)
    test_final_df = pd.concat([encoder_test_df, test_filled_df], axis=1)
    
    X_train = train_final_df.drop(['price_CHF'],axis=1).to_numpy()
    y_train = train_final_df['price_CHF'].to_numpy()
    X_test = test_final_df.to_numpy()

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions

    kernel = RationalQuadratic() * DotProduct()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    gpr.fit(X_train, y_train)
    
    y_pred = gpr.predict(X_test)
    
    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

