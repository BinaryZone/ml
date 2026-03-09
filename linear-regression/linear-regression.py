
# data
x_data = [1, 2, 3]
y_data = [2, 4, 6]

# param
w = 4

# model
def forward(x):
    return x * w

# loss
def cost(xs, ys):
    cost_value = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost_value += (y - y_pred) ** 2
    return cost_value / len(xs)

# gradient descent
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)

# train
for epoch in range(100):
    cost_val = cost(x_data, y_data)

    grad_val = gradient(x_data, y_data)

    w = w - (grad_val * 0.01)
    print('训练轮次： ', epoch, 'w=', w, 'loss=', cost_val)

