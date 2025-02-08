from NeuralNetwork import NeuralNetwork
from sys import argv
from colorama import Fore, init
init()


TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
SHAPE = [784, 80, 10]      # 784 input neurons, hidden neurons, 10 output neurons
MODEL = "model.npy"


def help():
    print(f"\33[2K{Fore.RED}Invalid number of arguments`{Fore.RESET}")
    print(f"Train model: python main.py [iterations] [alpha] [alpha_decay]")
    print(f"Run image: python main.py [model] [image]")
    print(f"Run test data: python main.py [model]")


try:
    match len(argv):
        case 2:
            if argv[1] == "help" or argv[1] == "--help" or argv[1] == "-h":
                help()
                exit(0)
            
            # Get accuracy on the TEST_FILE
            neural_network = NeuralNetwork()
            neural_network.load_model(argv[1])
            neural_network.test(TEST_FILE)

        case 3:
            # Predict the number in the image
            neural_network = NeuralNetwork()
            neural_network.load_model(argv[1])
            neural_network.predict(argv[2])
            
        case 4:
            # Train the model and save it
            iterations = int(argv[1])
            alpha = float(argv[2])
            alpha_decay = float(argv[3])
            neural_network = NeuralNetwork(SHAPE)
            neural_network.load_data(TRAIN_FILE)
            neural_network.int_random()
            neural_network.gradient_descent(iterations, alpha, alpha_decay)
            neural_network.save_model(MODEL)
            
        case _:
            help()
            exit(1)
            
except FileNotFoundError as e:
    print(f"{Fore.RED}\33[2KFile {str(e).split(" ")[-1]} not found{Fore.RESET}")
    exit(1)

except ValueError as e:
    print(f"{Fore.RED}\33[2KInvalid argument {str(e).split(" ")[-1]}{Fore.RESET}")
    exit(1)