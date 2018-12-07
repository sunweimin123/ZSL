# coding:utf-8
import torch




torch.manual_seed(1)

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)


def save():
    net = torch.nn.Sequential(torch.nn.Linear(1,10),torch.nn.ReLU(),torch.nn.Linear(10,1))
    optimizer = torch.optim.SGD(net.parameters(),lr = 0.01)
    loss_func = torch.nn.MSELoss()

    for i in range(100):
        prediction = net(x)
        loss=loss_func(prediction,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(net,'net.pkl')
    torch.save(net.state_dict(),'net1.pkl')

def loadGraph():
    net = torch.load('net.pkl')


def loadWeight():
    net = torch.nn.Sequential(torch.nn.Linear(1,10),torch.nn.ReLU,torch.nn.Linear(10,1))
    net.load_state_dict('net1.pkl')