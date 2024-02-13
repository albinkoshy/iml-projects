import pandas as pd
import numpy as np

df_train = pd.read_csv("./train.csv")
df_test = pd.read_csv("./test.csv")
df_sample = pd.read_csv("./sample.csv")

# Create to column in df_test with the average of x1 to x10
df_test['y'] = df_test.loc[:,'x1':'x10'].mean(axis=1)

# Save only the Id and y columns to a new dataframe
df_result = df_test[['Id', 'y']]
df_result.head()

# index to false, otherwise the index column will be included
df_result.to_csv("./result.csv", index=False)