import torch
from torch import nn

# Class could be any model name, arbitrary
class LogisticRegression(nn.Module): 

    def __init__(self, input):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input, 1), # Linear(input, output)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


# Now since the model is ready. Let's use it
model = LogisticRegression(16) # This will take 16 input parameters

criterion = nn.BCELoss() # This is the loss function

optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Defining an optimizer

# Now training the model for some epochs -> n_epochs
n_epochs = 10
x = [] # Some data
y = [] # some label

for _ in range(n_epochs):
    # Pass the value of X to the model to get the predictions
    y_pred = model(x)  
    # Check the error in the predicted output and the actual output
    loss = criterion(y_pred, y)

    # Backpropogate to update the weights to minimize the error

    # -> This is to make the gradients zero for every epoch as it can retain the values
    # from pervious epoch
    optimizer.zero_grad() 
    loss.backward() # calculate the weights to be updated for every neurons
    optimizer.step() # update the weights


# **** Done! Model is trained now ****
    