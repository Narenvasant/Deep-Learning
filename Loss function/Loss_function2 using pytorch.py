import torch
import matplotlib.pyplot as plt

# self defined loss function

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

# Creating random Tensors to hold inputs and outputs
x= torch.empty(N,D_in).normal_(mean=0,std=1)

y=torch.empty(N,D_out).normal_(mean=0,std=1)

model = torch.nn.Sequential(
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


model = TwoLayerNet(D_in, H, D_out)
# loss_fn = my_loss()
lr = 1e-3

# Initialising the batch size to input size to obtain Batch gradient descent for the entire input vector
batch_size = N
n_batches = int(len(x) / batch_size)
loss_arr = []
epochs = []

for t in range(1000):

    epochs.append(t)

    for batch in range(n_batches):

        batch_X, batch_y = x[batch * batch_size:(batch + 1) * batch_size, ], y[batch * batch_size:( batch + 1) * batch_size, ]
        y_pred = model(x)
        weights = []

        model.zero_grad()
        loss = my_loss(y_pred, y, x)
        loss_arr.append(loss)
        # print(loss)
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad
        
plt.plot(loss_arr,color='blue', label='loss function')
plt.legend(loc='upper right')
plt.xlabel("epochs")
plt.ylabel("error")
plt.show()