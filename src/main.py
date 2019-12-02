import warnings
import os

from src import manager
from src.adversarial_examples_generator.none_generator import NoneGenerator


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == "__main__":
    manager.execute_elaboration(NoneGenerator, 1000, True)
