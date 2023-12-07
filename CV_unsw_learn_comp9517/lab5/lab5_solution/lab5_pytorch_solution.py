import torch
import pandas as pd
import matplotlib.pyplot as plt

# Load data(do not change)
data = pd.read_csv("src/mnist_train.csv")
train_data = data[:2000]
test_data = data[2000:2500]

train_label = train_data['label']
test_label = test_data['label']

train_data = train_data.drop(["label"], axis = 1)
train_data = train_data.to_numpy() / 255
test_data = test_data.drop(["label"], axis = 1)
test_data = test_data.to_numpy() / 255


train_label = torch.tensor(train_label)
test_label = torch.tensor(test_label.to_numpy())

train_data = torch.tensor(train_data, dtype = torch.float32).view(-1, 1, 28, 28)
test_data = torch.tensor(test_data, dtype = torch.float32).view(-1, 1, 28, 28)

# Define your model here
class SmallNetWork(torch.nn.Module):
    def __init__(self):
        super(SmallNetWork, self).__init__()
        
        # conv layers: feature extractor
        # using nn.Sequential can concate layer together and more less code
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size = 5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10, 20, kernel_size = 5),
            torch.nn.Dropout2d(0.5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU()
        )
        
        # fc layers: classifier
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        # flatten the final output of conv_layers
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        # softmax can output log probability of each potential classes
        return torch.nn.functional.log_softmax(x, dim = 1)


model = SmallNetWork()
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = torch.nn.CrossEntropyLoss()

def PlotLearningCurve(epoch, trainingloss, testingloss):
    plt.plot(range(epoch), trainingloss, 'b', range(epoch), testingloss, 'r')
    plt.title('learning curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training loss', 'testing loss'])
    plt.show()

trainingloss = []
testingloss = []
epochs = 100
for epoch in range(1, epochs + 1):
    model.train()
    y_pred = model(train_data)
    loss = criterion(y_pred, train_label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    trainingloss += loss.item(),
    with torch.no_grad():
        model.eval()
        pred = model(test_data)
        loss = criterion(pred, test_label)
        testingloss += loss.item(),
        if epoch % 10 == 0:
            _, pred = torch.max(pred, dim = 1)
            acc = round((test_label == pred).type(torch.float32).mean().item() * 100, 2)
            print('Epoch:', epoch, 'test Accuracy:', acc)
        
PlotLearningCurve(epochs, trainingloss, testingloss)
