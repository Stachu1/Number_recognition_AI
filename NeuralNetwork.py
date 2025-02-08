import numpy as np, pandas as pd, time
from PIL import Image, ImageOps
from colorama import Fore, init
init()



class NeuralNetwork:
    def __init__(self, shape=[784, 80, 10]):
        self.shape = shape
        self.layers_count = len(shape) - 1
        self.input_layer = None
        self.label = None
        self.net = {}
        

    def load_data(self, file_name):
        print(f"{Fore.YELLOW}\33[2K Loading training data...{Fore.RESET}", end="\r")
        data = pd.read_csv(file_name)
        data = np.array(data)
        m, n = data.shape
        np.random.shuffle(data)

        data_dev = data[0:m].T
        self.label = data_dev[0]
        x = data_dev[1:n]
        self.input_layer = x / 255


    def int_random(self):
        print(f"{Fore.YELLOW}\33[2K Initialising neural network...{Fore.RESET}", end="\r")
        for l in range(self.layers_count):
            weight = np.random.rand(self.shape[l+1], self.shape[l]) - 0.5
            bias = np.random.rand(self.shape[l+1], 1) - 0.5
            self.net[f"w{l}"] = weight
            self.net[f"b{l}"] = bias
    
    
    def save_model(self, file_name):
        print(f"{Fore.YELLOW}\33[2K Saving model...{Fore.RESET}", end="\r")
        np.save(file_name, {"net": self.net, "shape": self.shape})
        print(f"{Fore.GREEN}\33[2KModel saved{Fore.RESET}")


    def load_model(self, file_name):
        data = np.load(file_name, allow_pickle=True).item()
        self.net = data["net"]
        self.shape = data["shape"]
        self.layers_count = len(self.shape) - 1


    def relu(self, v):
        return np.maximum(0, v)


    def derivative_relu(self, v):
        return v > 0


    def soft_max(self, v):
        a = np.exp(v) / sum(np.exp(v))
        return a


    def one_hot(self, v):
        one_hot = np.zeros((v.size, v.max() + 1))
        one_hot[np.arange(v.size), v] = 1
        one_hot = one_hot.T
        return one_hot


    def forward_propagation(self, input_layer):
        layers = []
        activated_layers = []
        layer = input_layer
        for l in range(self.layers_count):
            layer = self.net[f"w{l}"].dot(layer) + self.net[f"b{l}"]
            if l == self.layers_count - 1:
                return self.soft_max(layer), layers, activated_layers
            layers.append(layer)
            layer = self.relu(layer)
            activated_layers.append(layer)


    def backward_propagation(self, layers, activated_layers, output_layer):
        m = self.label.size
        one_hot_labels = self.one_hot(self.label)
        d_weights = []
        d_biases = []
        d_layer = output_layer - one_hot_labels     # Delta for last layer
        for l in range(self.layers_count-1, 0, -1):            # Delta for hidden layers
            d_weights.append(1 / m * d_layer.dot(activated_layers[l-1].T))
            d_biases.append(1 / m * np.sum(d_layer))
            d_layer = self.net[f"w{l}"].T.dot(d_layer) * self.derivative_relu(layers[l-1])

        # Delta for input layer
        d_weights.append(1 / m * d_layer.dot(self.input_layer.T))
        d_biases.append(1 / m * np.sum(d_layer))
        d_weights.reverse()
        d_biases.reverse()
        return d_weights, d_biases


    def update_values(self, d_weights, d_biases, alpha):
        for l in range(self.layers_count):
            self.net[f"w{l}"] = self.net[f"w{l}"] - alpha * d_weights[l]
            self.net[f"b{l}"] = self.net[f"b{l}"] - alpha * d_biases[l]


    def get_predictions(self, output_layer):
        return np.argmax(output_layer, 0)


    def get_accuracy(self, predictions):
        return np.sum(predictions == self.label) / self.label.size


    def gradient_descent(self, iterations=100, alpha=0.2, alpha_decay=0.99):
        print(f"{Fore.YELLOW}\33[2K Training...{Fore.RESET}", end="\r")
        time_start = time.time()
        for i in range(iterations):
            output_layer, layers, activated_layers = self.forward_propagation(self.input_layer)
            d_weights, d_biases = self.backward_propagation(layers, activated_layers, output_layer)
            self.update_values(d_weights, d_biases, alpha)

            accuracy = self.get_accuracy(self.get_predictions(output_layer))
            elapsed = time.time() - time_start
            eta = elapsed * iterations / (i+1) - elapsed
            alpha = alpha * alpha_decay if i % 10 == 0 else alpha
            print(f"{Fore.MAGENTA}\33[2K Iteration: {Fore.RESET}{i+1}   {Fore.GREEN}Accuracy: {Fore.RESET}{(accuracy*100):.2f}%   {Fore.CYAN}ETA: {Fore.RESET}{eta:.0f}s", end="\r")
        print(f"{Fore.MAGENTA}\33[2KIteration: {Fore.RESET}{iterations}   {Fore.GREEN}Accuracy: {Fore.RESET}{(accuracy*100):.2f}%   {Fore.CYAN}Elapsed: {Fore.RESET}{elapsed:.0f}s")


    def test(self, file_name):
        print(f"{Fore.YELLOW}\33[2K Testing...{Fore.RESET}", end="\r")
        data = pd.read_csv(file_name)
        data = np.array(data)
        m, n = data.shape
        np.random.shuffle(data)

        data_dev = data[0:m].T
        self.label = data_dev[0]
        x = data_dev[1:n]
        input_layer = x / 255
        output_layer, _, _ = self.forward_propagation(input_layer)
        accuracy = self.get_accuracy(self.get_predictions(output_layer))
        print(f"{Fore.GREEN}Accuracy: {Fore.RESET}{(accuracy*100):.2f}%")


    def predict(self, file_name):
        img = Image.open(file_name).resize((28,28)).convert("L")
        img = ImageOps.invert(img)
        threshold = max(img.getpixel((0,0)), img.getpixel((27,0)), img.getpixel((0,27)), img.getpixel((27,27)))
        img = img.point(lambda p: 0 if p < threshold else p)
        input_layer = np.reshape(np.asarray(img), (784, 1)) / 255
        output_layer, _, _ = self.forward_propagation(input_layer)
        for index, value in enumerate(output_layer):
            print(f"{index}: {round(value[0]*100, 2)}%")
        print(f"Predicted: {Fore.GREEN}{self.get_predictions(output_layer)[0]}{Fore.RESET}")



if __name__ == "__main__":
    TRAIN_FILE = "data/train.csv"
    TEST_FILE = "data/test.csv"
    SHAPE = [784, 80, 10]      # 784 input neurons, hidden neurons, 10 output neurons
    MODEL = "model.npy"
    neural_network = NeuralNetwork(SHAPE)
    neural_network.load_data(TRAIN_FILE)
    neural_network.int_random()
    neural_network.gradient_descent()
    neural_network.save_model(MODEL)
    neural_network.load_model(MODEL)
    neural_network.test(TEST_FILE)
