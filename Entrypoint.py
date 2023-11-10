from gan.dcgan.Execution import run as run_dcgan
import sys


MODELS = {
    "dcgan": run_dcgan
}


def no_model_found_handler(_):
    print("No model found")
    print("Valid models: ")
    for model in MODELS:
        print(model)


def start_model(model, command_arguments):
    MODELS.get(model, no_model_found_handler)(command_arguments)


if __name__ == "__main__":
    start_model(sys.argv[1], sys.argv[2:])
