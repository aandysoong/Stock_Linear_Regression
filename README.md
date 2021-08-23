# Stock_Linear_Regression
## Introduction

In this repository, I will be explaining the basics of linear regression and apply it to some stock data. Although the prediction model will not be completely accurate (since stock is so unpredictable), it can provide a good sense of how companies have been doing long term. Overall though, this project will help explain linear regression and how to use it in Python.

## Linear Regression
Linear Regression, which will be referred from now on as LR, is a type of machine learning which results in the computer finding the best possible linear graph for a set of data. After a linear graph/equation is formed, the graph can be used for predictions, although it is not always precise or accurate. However, it is useful in locating general trends and making simple predictions. Since the process itself is long and complicated, I will break it down into many parts.

### Data and Training
As mentioned previously, LR is when a computer "learns" the data it is given and makes a general linear graph about it. The data here must be a two dimensional set.

Example:

x = [1,2,3]

y = [2,4,6]

These values, when plotted on a 2d graph, generate the points (1,2), (2,4), and (3,6). They all lie on the y = 2x graph.

![스크린샷 2021-08-22 오후 4 56 29](https://user-images.githubusercontent.com/70020467/130347073-efeae305-5fa8-4738-b289-55877fbc3ec5.png)

Now, if we want the computer to learn that these 3 points lie on y=2x graph, we must provide the data to the computer. The data provided here is called the **training set**. Specifically, the x data is called the "x train" and the y data the "y train".


### Setting up a model/hypothesis

When the computer takes in the training set, it needs to find the best fit equation for the points. In order to the computer to do this, we need to provide the computer with a basic equation for it to work with. This is often referred to as a **model or hypothesis**.

Example:

y = wx + b

(also expressed as H(x) = wx + b, with H for hypothesis)

This equation is just like a normal linear graph. However, the coefficient of x and the y intercept is left as variables w and b. w stands for weight and b for bias. These are just terms in LR and machine learning in general. But to understand it simply, w is the slope of an equation and b the y intercept (at least in LR).

### Cost function

The cost function, also called the loss, error, or objective function, is the computer's way of evaluating whether a hypothesis that it created is good or bad. 

Example:

Data points: (1,2), (2,4), (3,6)

Hypothesis/model: y = x

In this case, the computer is guessing that the y = x model is the best fit line for the given data points. Obviously, this created model is not correct. But in order for the computer to check the "correctness", it subtracts the predicted value from the actual value, squares the difference, and adds them all together. To explain further, when the computer used a y=x equation,

|  x value  |  1  |  2  |  3  |
|---|---|---|---|
|  actual y value  |  2  |  4  |  6  |
|  predicted value  |  1  |  2  |  3  |
|  difference  |  1  |  2  |  3  |
|  diff^2  |  1  |  4  |  9  |

cost = 1 + 4 + 9 = 14

You may be wondering, why square the difference and then add them together? This is because somtimes, the difference may be a negative number. In that case, simply adding the differences may cause some problems. Therefore, to recap, the cost function is the sum of (actual - predicted)^2 for every x training set value. In a mathematical equation, it is 

![cost function](https://humanunsupervised.github.io/humanunsupervised.com/topics/images/lesson2/14.png)

Image from: https://humanunsupervised.github.io/humanunsupervised.com/topics/images/lesson2/14.png

### Gradient Descent

The objective of the code is trying to minimize the cost of a function. In order to minimize, they go through a process called **gradient descent**, a cycle which changes the weight and bias of a function. In order to understand gradient descent, we first need to visualize a W-Cost graph.

Graph visualized in this link: https://www.desmos.com/calculator/3gkkybqudo

As shown in the graph, we can express W and Cost in a quadratic graph (we will exclude b for now). The purpose of the computer is now to try find the lowest value of the Cost or the minimum of the quadratic function. And the minimum of a polynomial is found by differentiation. (This is calculus math so it may be hard to understand). Anyways, the computer repeats a process of plugging in a certain W value and finding the cost, each time receiving a smaller cost value. You may be wondering, how does the computer decide which number to plug in every single time? Well, that is determined by the Learning Rate (lr) and the optimizer. To put it simply, the optimizer decides which way the gradient descent should occur and how the computer should repeat plugging in W values. The learning rate tell the computer how much it should alter its W values each time. The b value is also calculated at the same time so that the user gets the best W and b values for a set of points in the very end.

(The optimizer and learning rate are hard concepts but think of it like a ball moving down a slope/graph)

![gradient descent](https://wikidocs.net/images/page/21670/%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95.PNG)
Image from: https://wikidocs.net/images/page/21670/%EA%B2%BD%EC%82%AC%ED%95%98%EA%B0%95%EB%B2%95.PNG

Here, the graph is the W-Cost relation and the ball is moving downwards to the minimum. The optimizer tells the ball which way to go and the learning rate how much to go each time. With each repetition, the ball will slowly move downward.

Gradient Descent is more complicated than just this though, so I recommend that you go look at some other sources if interested. However, it is extremely hard and I often struggle to understand as well, so I will leave my explanation to this.

## Actual Code

Finally, we can move all this theory into action and create a linear regression model in python. For this, we will import a tool called PyTorch. It is a commonly used tool for machine learning. Specifically, we need the nn module. The **nn module** is a Regression model which makes life much easier for coders.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Since this project is dealing with stock data, we need to import it. I will be using Yahoo Finance. 

```python
import yfinance as yf
```
I will also import matplotlib (graphing for python) just for later.

```python
import matplotlib.pyplot as plt
```

Now, from Yahoo Finance, we want to get some data about a specific stock. For this model, I want a stock that has had a clear rise or decline in recent times. So I selected Tesla. To import Tesla stock data, we use Ticker, a feature in Yahoo Finance.

```python
tsla=yf.Ticker('tsla').history(period='365d')['Close']
```

The 'tsla' part is where you can type in the name of a stock you want (make sure you use the official stock name). "history(period='365d')" allows us to look at data from the past year. "['Close']" is used to import the Closing price of a stock. (The closing price is the last value of how much a stock was bought/sold for that day) With this data, we can now set up the training sets.

Just to recap, the x train will be the dates and the y train the daily stock prices.

The method in which we will create the training sets will be by going through each value in the dataset we acquired and adding it to a list called x_list or y_list. Then, we will put the list into a Tensor. The reason we put it into a tensor is because that makes the training sets compatible to the nn module. the x_list and y_list will be used later on during graphing as well.

```python
y_list=[]
for i in tsla:
    y_list.append([i]) # y_list completed
y_train = torch.FloatTensor(y_list) # move into a tensor

x_list=[]
for i in range(1,len(tsla)+1):
    x_list.append([i])
x_train = torch.FloatTensor(x_list)
# repeat the process for x data as well
```

Now, we set up the nn module. 

```python
model = nn.Linear(1,1)
```

The "Linear" represents that we are conducting a linear regression model. The (1,1) represents that we have one set of x data and one set of y data
