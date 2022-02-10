import math
def forward_prop():
    return None
y_true = 1.0
def back_prop():
    return None

LEARNING_RATE = 0.1 #学習率(lr)

def update_param(grads_w, grads_b, lr = 0.1):
    return None

#訓練処理
y_pred, cached_outs, cached_sums = forward_prop(cache_mode=True)
grads_w, grads_b = back_prop(y_true, cached_outs, cached_sums)
weights, biases = update_param(grads_w, grads_b, LEARNING_RATE)

print('予測値:{y_pred}')
print('正解地:{y_true}')

layers = [
    2, #input layer 
    3, #hidden layer
    1] #output layer Node

weights = [
    [[0.0, 0.0],[0.0, 0.0],[0.0, 0.0]], #input Layer -> hidden Layer
    [[0.0, 0.0, 0.0]] #hidden Layer -> output Layer
]

biases = [
    [0.0, 0.0, 0.0], #hidden Layer1
    [0.0] # output Layer
]

model = (layers, weights, biases)
x = [0.05, 0.1]

def summation(x, weights, bias):
    linear_sum = 0.0
    
    for x_i, w_i in zip(x, weights):
        linear_sum += x_i * w_i
    linear_sum += bias
    return linear_sum

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_der(x):
    output = sigmoid(x)
    return output * (1.0 - output)

def identity(x):
    return x

def identity_der(x):
    return 1.0

w = [0.0, 0.0]
b = 0.0

next_x = x

node_sum = summation(next_x, w, b)

is_hidden_layer = True
if is_hidden_layer:
    node_out = sigmoid(node_sum)
else:
    node_out = identity(node_sum)

