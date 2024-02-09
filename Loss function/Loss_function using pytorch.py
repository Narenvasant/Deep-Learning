import torch
import matplotlib.pyplot as plt


def my_loss(actual, predicted, inputs):
    m = torch.nn.Softmax(dim=0)
    output = m(inputs)
    a = torch.sum(output, dim=1)
    weighted_inputs = torch.diag_embed(a, offset=0)

    t1 = torch.transpose((actual - predicted), 0, 1)
    t2 = actual - predicted

    t = t1 * weighted_inputs * t2

    loss = torch.trace(t)
    return 0.5 * loss


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1000, 6, 3, 1

# Create random Tensors to hold inputs and outputs

x= torch.empty(N,D_in).normal_(mean=0,std=1)

y=torch.empty(N,D_out).normal_(mean=0,std=1)

model1 = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax())
model2 = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax())


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


model1 = TwoLayerNet(D_in, H, D_out)
model2 = TwoLayerNet(D_in, H, D_out)

lr = 1e-2
optimizer1 = torch.optim.SGD(model1.parameters(), lr)
optimizer2 = torch.optim.SGD(model2.parameters(), lr)

my_loss_MSE = torch.nn.MSELoss()
loss_arr_MSE = []
loss_arr = []

#Weighted mean square error loss function

for t in range(1000):
    y_pred = model1(x)

    model1.zero_grad()

    loss = my_loss(y_pred, y, x)

    loss_arr.append(loss)

    loss.backward()

    #with torch.no_grad():

        #for param in model1.parameters():

           #param -= lr * param.grad

    optimizer1.step()

line1= plt.plot(loss_arr,color='red', label='Weighted MSE loss function')
plt.legend(loc='upper right')
plt.show()

#Mean square error loss function

for t in range(1000):
    y_pred = model2(x)

    model2.zero_grad()
    loss_mse = my_loss_MSE(y_pred, y)

    loss_arr_MSE.append(loss_mse)

    loss_mse.backward()

    #with torch.no_grad():

        #for param in model2.parameters():
            #param -= lr * param.grad

    optimizer2.step()

line2=plt.plot(loss_arr_MSE,color='blue', label='MSE loss function')
plt.xlabel("epochs")
plt.ylabel("error")

plt.legend(loc='upper right')
plt.show()