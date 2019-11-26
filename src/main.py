from src import manager
from src.adversarial_examples_generator.none_generator import NoneGenerator


if __name__ == "__main__":
    manager.execute_elaboration(NoneGenerator, 1000, True)
