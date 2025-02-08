import numpy as np, pandas as pd, time, sys
from PIL import Image, ImageOps
from colorama import Fore, init
init()




TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
SHAPE = [784, 128, 64, 10]      # 784 input nodes, 100 hidden nodes, 10 output nodes

MODEL = "model.npy"
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
        net[f"w{l}"] = weight
        net[f"b{l}"] = bias
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
        layer = net[f"w{l}"].dot(input_layer) + net[f"b{l}"]
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
    d_layer = output_layer - one_hot_labels     # Delta for last layer
    for l in range(LAYERS-1, 0, -1):            # Delta for hidden layers
        d_weights.append(1 / m * d_layer.dot(activated_layers[l-1].T))
        d_biases.append(1 / m * np.sum(d_layer))
        d_layer = net[f"w{l}"].T.dot(d_layer) * derivative_relu(layers[l-1])
    
    # Delta for input layer
    d_weights.append(1 / m * d_layer.dot(input_layer.T))
    d_biases.append(1 / m * np.sum(d_layer))
    d_weights.reverse()
    d_biases.reverse()
    return d_weights, d_biases


def update_values(net, d_weights, d_biases, alpha):
    for l in range(LAYERS):
        net[f"w{l}"] = net[f"w{l}"] - alpha * d_weights[l]
        net[f"b{l}"] = net[f"b{l}"] - alpha * d_biases[l]
    return net


def get_predictions(output_layer):
    return np.argmax(output_layer, 0)


def get_accuracy(predictions, label):
    return np.sum(predictions == label) / label.size


def gradient_descent(input_layer, label, iterations, alpha):
    with open("training.log", "w") as f:
        net = init_params()
        time_start = time.time()
        for i in range(iterations):
            output_layer, layers, activated_layers = forward_propagation(net, input_layer)
            d_weights, d_biases = backward_propagation(net, layers, activated_layers, input_layer, output_layer, label)
            net = update_values(net, d_weights, d_biases, alpha)
            accuracy = get_accuracy(get_predictions(output_layer), label)
            elapsed = time.time() - time_start
            eta = elapsed * iterations / (i+1) - elapsed
            f.write(f"{accuracy}\n")
            if (i % 1) == 0:
                print(f"{Fore.MAGENTA}Iteration: {Fore.RESET}{i+1}   {Fore.GREEN}Accuracy: {Fore.RESET}{(accuracy*100):.2f}%   {Fore.CYAN}ETA: {Fore.RESET}{eta:.2f}s      ", end="\r")
        print(f"{Fore.MAGENTA}Iteration: {Fore.RESET}{iterations}   {Fore.GREEN}Accuracy: {Fore.RESET}{(accuracy*100):.2f}%   {Fore.CYAN}Elapsed: {Fore.RESET}{elapsed:.2f}s      ")
    return net



if __name__ == "__main__":
    try:
        match len(sys.argv):
            case 1:
                # Get accuracy on the TEST_FILE
                input_layer, label = get_data(TEST_FILE)
                net = np.load(MODEL, allow_pickle=True).item()
                output_layer, layers, activated_layers = forward_propagation(net, input_layer)
                accuracy = get_accuracy(get_predictions(output_layer), label)
                print(f"{Fore.GREEN}Accuracy: {Fore.RESET}{(accuracy*100):.2f}%")

            case 2:
                # Predict the number in the image
                img = Image.open(sys.argv[1]).resize((28,28)).convert("L")
                img = ImageOps.invert(img)
                threshold = max(img.getpixel((0,0)), img.getpixel((27,0)), img.getpixel((0,27)), img.getpixel((27,27)))
                img = img.point(lambda p: 0 if p < threshold else p)
                input_layer = np.asarray(img)
                input_layer = np.reshape(input_layer, (784, 1))
                input_layer = input_layer / 255
                net = np.load(MODEL, allow_pickle=True).item()
                output_layer, layers, activated_layers = forward_propagation(net, input_layer)
                for index, value in enumerate(output_layer):
                    print(f"{index}: {round(value[0]*100, 2)}%")
                print(f"Predicted: {Fore.GREEN}{get_predictions(output_layer)[0]}{Fore.RESET}")

            case 3:
                # Train the model and save it
                input_layer, label = get_data(TRAIN_FILE)
                iterations = int(sys.argv[1])
                alpha = float(sys.argv[2])
                net = gradient_descent(input_layer, label, iterations, alpha)
                np.save(MODEL, net)
                print(f"{Fore.GREEN}Model saved{Fore.RESET}")

            case _:
                print(f"{Fore.RED}Invalid number of arguments`{Fore.RESET}")
                print(f"Train model: python main.py [iterations] [alpha]")
                print(f"Run image: python main.py [image_path]")
                print(f"Run test data: python main.py")
                exit(1)
    except FileNotFoundError as e:
        print(f"{Fore.RED}File {str(e).split(" ")[-1]} not found{Fore.RESET}")
        exit(1)