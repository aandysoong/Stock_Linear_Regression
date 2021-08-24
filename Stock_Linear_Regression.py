import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yfinance as yf
# all the neccessary inputs
torch.manual_seed(1)

# get the data that you want
tsla=yf.Ticker('tsla').history(period='365d')['Close']

# set up the model and the training set 
model = nn.Linear(1,1)

y_list=[]
for i in tsla:
    y_list.append([i])
y_train = torch.FloatTensor(y_list)

x_list=[]
for i in range(1,len(tsla)+1):
    x_list.append([i])
x_train = torch.FloatTensor(x_list)

optimizer = torch.optim.SGD(model.parameters(), lr=0.00001) 


# actual learning process
nb_epochs = 100000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)

    # finding the cost
    cost = F.mse_loss(prediction, y_train) 

    # reset the gradient
    optimizer.zero_grad()
    cost.backward()
    # update W and b
    optimizer.step()

    # print to show
    if epoch % 10000 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
        print(list(model.parameters()))

new_var =  torch.FloatTensor([[3]]) 
pred_y = model(new_var) 
print("predicted value for 3 :", pred_y) 

# data for plotting
y_plot_list=[]
w_final = float(list(model.parameters())[0].data.numpy()[0][0])
b_final = float(list(model.parameters())[1].data.numpy()[0])

for i in x_list:
    y_plot_list.append(float(model(torch.FloatTensor([[i]]))))

# actual plotting
plt.plot(x_train,y_plot_list)
plt.plot(x_train, y_train)
plt.xlabel('Days')
plt.ylabel('Value')
plt.title('TSLA')
plt.show()
