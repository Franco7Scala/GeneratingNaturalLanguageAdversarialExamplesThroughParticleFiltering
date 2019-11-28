import warnings

from src import manager
from src.adversarial_examples_generator.none_generator import NoneGenerator


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    manager.execute_elaboration(NoneGenerator, 1000, True)
