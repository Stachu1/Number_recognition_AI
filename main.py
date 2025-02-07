import numpy as np, pandas as pd, time, sys
from PIL import Image, ImageOps
from colorama import Fore, init
init()




TRAIN_FILE = "data/mnist_train.csv"
TEST_FILE = "data/mnist_test.csv"
SHAPE = [784, 80, 20, 10]      # 784 input nodes, 100 hidden nodes, 10 output nodes
LAYERS = len(SHAPE) - 1



def get_data(file_name=TRAIN_FILE):
    data = pd.read_csv(file_name)
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)
    
    data_dev = data[0:m].T
    y = data_dev[0]
    x = data_dev[1:n]
    x = x / 255
    _, m = x.shape
    return x, y


def init_params():
    net = {}
    for l in range(LAYERS):
        weight = np.random.rand(SHAPE[l+1], SHAPE[l]) - 0.5
        bias = np.random.rand(SHAPE[l+1], 1) - 0.5
        net[f"weight{l}"] = weight
        net[f"bias{l}"] = bias
    return net


def relu(v):
    return np.maximum(0, v)


def derivative_relu(v):
    return v > 0


def soft_max(v):
    a = np.exp(v) / sum(np.exp(v))
    return a


def one_hot(v):
    one_hot = np.zeros((v.size, v.max() + 1))
    one_hot[np.arange(v.size), v] = 1
    one_hot = one_hot.T
    return one_hot


def forward_propagation(net, input_layer):
    layers = []
    activated_layers = []
    for l in range(LAYERS):
        layer = net[f"weight{l}"].dot(input_layer) + net[f"bias{l}"]
        if l == LAYERS - 1:
            return soft_max(layer), layers, activated_layers
        input_layer = relu(layer)
        layers.append(layer)
        activated_layers.append(input_layer)


def backward_propagation(net, layers, activated_layers, input_layer, output_layer, label):
    m = label.size
    one_hot_labels = one_hot(label)
    d_weights = []
    d_biases = []
    d_layer = output_layer - one_hot_labels
    for l in range(LAYERS-1, 0, -1):
        d_weights.append(1 / m * d_layer.dot(activated_layers[l-1].T))
        d_biases.append(1 / m * np.sum(d_layer))
        
        d_layer = net[f"weight{l}"].T.dot(d_layer) * derivative_relu(layers[l-1])
        
    
    d_weights.append(1 / m * d_layer.dot(input_layer.T))
    d_biases.append(1 / m * np.sum(d_layer))
    d_weights.reverse()
    d_biases.reverse()
    return d_weights, d_biases
    
    # d_layer2 = output_layer - one_hot_labels
    
    # d_weight2 = 1 / m * d_layer2.dot(layer1_activated.T)
    # d_bias2 = 1 / m * np.sum(d_layer2)
    
    # d_layer1 = weight2.T.dot(d_layer2) * derivative_relu(layer1)
    
    # d_weight1 = 1 / m * d_layer1.dot(input_layer.T)
    # d_bias1 = 1 / m * np.sum(d_layer1)
    # return d_weight1, d_bias1, d_weight2, d_bias2


def update_values(net, d_weights, d_biases, alpha):
    for l in range(LAYERS):
        net[f"weight{l}"] = net[f"weight{l}"] - alpha * d_weights[l]
        net[f"bias{l}"] = net[f"bias{l}"] - alpha * d_biases[l]
    return net


def get_predictions(output_layer):
    return np.argmax(output_layer, 0)


def get_accuracy(predictions, label):
    return np.sum(predictions == label) / label.size


def gradient_descent(input_layer, label, iterations, alpha):
    time_start = time.time()
    with open("accu.log", "w") as f:
        net = init_params()
        for i in range(iterations):
            output_layer, layers, activated_layers = forward_propagation(net, input_layer)
            d_weights, d_biases = backward_propagation(net, layers, activated_layers, input_layer, output_layer, label)
            net = update_values(net, d_weights, d_biases, alpha)
            accuracy = get_accuracy(get_predictions(output_layer), label)
            eta = (time.time() - time_start) * iterations / (i+1) -(time.time() - time_start)
            f.write(f"{accuracy}\n")
            if (i % 1) == 0:
                print(f"{Fore.YELLOW}Iteration: {Fore.RESET}{i+1}   {Fore.GREEN}Accuracy: {Fore.RESET}{(accuracy*100):.2f}%   {Fore.CYAN}ETA: {Fore.RESET}{eta:.2f}s      ", end="\r")
        print(f"{Fore.YELLOW}Iteration: {Fore.RESET}{iterations}   {Fore.GREEN}Accuracy: {Fore.RESET}{(accuracy*100):.2f}%")
    return net


if len(sys.argv) > 1:
    file_name = sys.argv[1]
    img = Image.open(file_name).resize((28,28)).convert("L")
    img = ImageOps.invert(img)
    img = img.point(lambda p: 0 if p < 150 else p)
    img.show()
    input_layer = np.asarray(img)
    input_layer = np.reshape(input_layer, (784, 1))
    input_layer = input_layer / 255
    net = np.load("model.npy")
    output_layer, layers, activated_layers = forward_propagation(net, input_layer)
    for index, value in enumerate(output_layer):
        print(f"{index}: {round(value[0]*100, 2)}%")
    

else:
    if input("\nDo you want to train a new model? (Y/n): ") == "n":
        input_layer, label = get_data(TRAIN_FILE)
        net = np.load("model.npy", allow_pickle=True).item()
        output_layer, layers, activated_layers = forward_propagation(net, input_layer)
        accuracy = get_accuracy(get_predictions(output_layer), label)
        print(f"{Fore.GREEN}Accuracy: {Fore.RESET}{(accuracy*100):.2f}%")
    else:
        input_layer, label = get_data(TEST_FILE)
        iterations = int(input("Iterations: "))
        alpha = float(input("Alpha: "))
        net = gradient_descent(input_layer, label, iterations, alpha)
        np.save("model.npy", net)