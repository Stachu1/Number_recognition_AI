import numpy as np, pandas as pd, time, sys
from PIL import Image
from colorama import Fore, init
init()




TRAIN_FILE = "data/mnist_train.csv"
TEST_FILE = "data/mnist_test.csv"
SHAPE = [784, 64, 10]


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
    weight1 = np.random.rand(SHAPE[1], SHAPE[0]) - 0.5
    bias1 = np.random.rand(SHAPE[1], 1) - 0.5
    weight2 = np.random.rand(SHAPE[2], SHAPE[1]) - 0.5
    bias2 = np.random.rand(SHAPE[2], 1) - 0.5
    return weight1, bias1, weight2, bias2


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


def forward_propagation(weight1, bias1, weight2, bias2, input_layer):
    layer1 = weight1.dot(input_layer) + bias1
    layer1_activated = relu(layer1)
    layer2 = weight2.dot(layer1_activated) + bias2
    output_layer = soft_max(layer2)
    return layer1, layer1_activated, layer2, output_layer


def backward_propagation(input_layer, layer1, layer1_activated, layer2, weight2, output_layer, label):
    m = label.size
    one_hot_labels = one_hot(label)
    d_layer2 = output_layer - one_hot_labels
    d_weight2 = 1 / m * d_layer2.dot(layer1_activated.T)
    d_bias2 = 1 / m * np.sum(d_layer2)
    d_layer1 = weight2.T.dot(d_layer2) * derivative_relu(layer1)
    d_weight1 = 1 / m * d_layer1.dot(input_layer.T)
    d_bias1 = 1 / m * np.sum(d_layer1)
    return d_weight1, d_bias1, d_weight2, d_bias2


def update_values(weight1, bias1, weight2, bias2, d_weight1, d_bias1, d_weight2, d_bias2, alpha):
    weight1 = weight1 - alpha * d_weight1
    bias1 = bias1 - alpha * d_bias1
    weight2 = weight2 - alpha * d_weight2
    bias2 = bias2 - alpha * d_bias2
    return weight1, bias1, weight2, bias2


def get_predictions(output_layer):
    return np.argmax(output_layer, 0)


def get_accuracy(predictions, label):
    return np.sum(predictions == label) / label.size


def gradient_descent(input_layer, label, iterations, alpha):
    time_start = time.time()
    with open("log.txt", "w") as f:
        weight1, bias1, weight2, bias2 = init_params()
        for i in range(iterations):
            layer1, layer1_activated, layer2, output_layer = forward_propagation(weight1, bias1, weight2, bias2, input_layer)
            d_weight1, d_bias1, d_weight2, d_bias2 = backward_propagation(input_layer, layer1, layer1_activated, layer2, weight2, output_layer, label)
            weight1, bias1, weight2, bias2 = update_values(weight1, bias1, weight2, bias2, d_weight1, d_bias1, d_weight2, d_bias2, alpha)
            accuracy = get_accuracy(get_predictions(output_layer), label)
            eta = (time.time() - time_start) * iterations / (i+1) -(time.time() - time_start)
            f.write(f"{accuracy}\n")
            if (i % 1) == 0:
                print(f"{Fore.YELLOW}Iteration: {Fore.RESET}{i+1}   {Fore.GREEN}Accuracy: {Fore.RESET}{(accuracy*100):.2f}%   {Fore.CYAN}ETA: {Fore.RESET}{eta:.2f}s      ", end="\r")
        print(f"{Fore.YELLOW}Iteration: {Fore.RESET}{iterations}   {Fore.GREEN}Accuracy: {Fore.RESET}{(accuracy*100):.2f}%")
    return weight1, bias1, weight2, bias2


if len(sys.argv) > 1:
    file_name = sys.argv[1]
    img = Image.open(file_name).resize((28,28)).convert("L")
    input_layer = np.asarray(img)
    input_layer = np.reshape(input_layer, (784, 1))
    input_layer = input_layer / 255
    model = np.load("model.npz")
    weight1 = model["name1"]
    bias1 = model["name2"]
    weight2 = model["name3"]
    bias2 = model["name4"]
    layer1, layer1_activated, layer2, output_layer = forward_propagation(weight1, bias1, weight2, bias2, input_layer)
    for index, value in enumerate(output_layer):
        print(f"{index}: {round(value[0]*100, 2)}%")
    

else:
    if input("\nDo you want to train new model?(Y/n): ") == "n":
        input_layer, label = get_data(TRAIN_FILE)
        model = np.load("model.npz")
        weight1 = model["name1"]
        bias1 = model["name2"]
        weight2 = model["name3"]
        bias2 = model["name4"]
        layer1, layer1_activated, layer2, output_layer = forward_propagation(weight1, bias1, weight2, bias2, input_layer)
        accuracy = get_accuracy(get_predictions(output_layer), label)
        print(f"{Fore.GREEN}Accuracy: {Fore.RESET}{(accuracy*100):.2f}%")
    else:
        input_layer, label = get_data(TEST_FILE)
        iterations = int(input("Iterations: "))
        alpha = float(input("Alpha: "))
        weight1, bias1, weight2, bias2 = gradient_descent(input_layer, label, iterations, alpha)
        np.savez("model.npz", name1=weight1, name2=bias1, name3=weight2, name4=bias2)