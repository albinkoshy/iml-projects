# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EMBEDDING_SIZE = 2048

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # TODO: define a transform to pre-process the images
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transforms)
    # Hint: adjust batch_size and num_workers to your PC configuration, so that you don't 
    # run out of memory
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=False,
                              pin_memory=True, num_workers=4)

    # TODO: define a model for extraction of the embeddings (Hint: load a pretrained model,
    #  more info here: https://pytorch.org/vision/stable/models.html)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    embeddings = []
    # Last layer of model
    embedding_size = EMBEDDING_SIZE # Dummy variable, replace with the actual embedding size once you 
    # pick your model
    num_images = len(train_dataset) # 10000
    embeddings = np.zeros((num_images, embedding_size)) # (10000, embedding_size)
    # TODO: Use the model to extract the embeddings. Hint: remove the last layers of the 
    # model to access the embeddings the model generates.

    # Remove the last layer of model
    model = torch.nn.Sequential(*list(model.children())[:-1])

    # Move model to gpu if gpu is available
    model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # temporarily sets all of the requires_grad flags to false
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            inputs, _ = data # no labels
            # Move data to gpu if gpu is available
            inputs = inputs.to(device)
            outputs = model(inputs)
            # first move data from gpu (if used) to cpu memory. Data consists of BATCH_SIZE samples. 
            output_np = outputs.cpu().numpy() # shape: (64, EMBEDDING_SIZE, 1, 1)
            # last batch has size 16
            output_np = np.reshape(output_np, (16 if inputs.shape[0] == 16 else 64, embedding_size))
            embeddings[i*train_loader.batch_size: (i+1)*train_loader.batch_size] = output_np
    
    np.save('dataset/embeddings_res50.npy', embeddings)


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="dataset/",
                                         transform=None)
    # Rename file names from '00000.jpg' to '00000'
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('dataset/embeddings_res50.npy') # (10000, EMBEDDING_SIZE)
    # TODO: Normalize the embeddings across the dataset
    #embeddings = (embeddings-embeddings.min())/(embeddings.max()-embeddings.min())
    embeddings = (embeddings - np.mean(embeddings))/np.std(embeddings)
    
    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

# Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 16):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader

# TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.fc1 = nn.Linear(3*EMBEDDING_SIZE, 3556)
        self.fc2 = nn.Linear(3556, 1536)
        self.fc3 = nn.Linear(1536, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64, 1)
    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        x = torch.sigmoid(x)
        return x

def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.to(device)
    model.train()
    n_epochs = 1
    losses = []
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # TODO: define a loss function, optimizer and proceed with training. Hint: use the part 
    # of the training data as a validation split. After each epoch, compute the loss on the 
    # validation split and print it out. This enables you to see how your model is performing 
    # on the validation data before submitting the results on the server. After choosing the 
    # best model, train it on the whole training data.
    for epoch in range(n_epochs):        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            y = y.unsqueeze(1)
            loss = loss_function(output, y.float())
            losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Epoch {}, Batch idx {}, loss {}'.format(
                   epoch, batch_idx, loss.item()))
    return model

def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch= x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('dataset/embeddings_res50.npy') == False):
        generate_embeddings()

    # load the training and testing data
    X, y = get_data(TRAIN_TRIPLETS)
    X_test, _ = get_data(TEST_TRIPLETS, train=False)

    # Create data loaders for the training and testing data
    train_loader = create_loader_from_np(X, y, train = True, batch_size=64, num_workers = 4)
    test_loader = create_loader_from_np(X_test, train = False, batch_size=2048, shuffle=False, num_workers = 4)

    # define a model and train it
    model = train_model(train_loader)
    
    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")
