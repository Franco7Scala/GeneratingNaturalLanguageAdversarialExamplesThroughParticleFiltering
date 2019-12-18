import time
import os
import numpy
import json

from os import path
from src.support import support


class AdversarialExampleGenerator:

    def __init__(self, model, level, verbose):
        self.name = self.__class__.__name__
        self.model = model
        self.level = level
        self.verbose = verbose
        self.configuration = self._read_configuration()

    def generate_adversarial_examples(self, examples_x, examples_y, verbose = False):
        support.colored_print("Generating Adversarial Examples with technique: {}...".format(self.name), "green", verbose)
        successful_perturbations = 0
        failed_perturbations = 0
        adversarial_examples = []
        sub_rate_list = []
        NE_rate_list = []
        time_to_delete = 0
        start_cpu = time.clock()
        adversarial_text_path = support.get_adversarial_text_path(self.model.phrase_manager.name, self.model.name, len(examples_x))
        adversarial_text_path_simple = adversarial_text_path[0:adversarial_text_path.rfind("/")]
        words_changed_path = support.get_changed_words_path(self.model.phrase_manager.name, self.model.name, len(examples_x))
        words_changed_path_simple = adversarial_text_path[0:words_changed_path.rfind("/")]
        if not os.path.exists(adversarial_text_path_simple):
            os.makedirs(adversarial_text_path_simple)

        file_adversarial_examples = open(adversarial_text_path, "a")
        if not os.path.exists(words_changed_path_simple):
            os.makedirs(words_changed_path_simple)

        file_changed_words = open(words_changed_path, "a")
        for index, text in enumerate(examples_x):
            sub_rate = 0
            NE_rate = 0
            # if model predicted correctly the example
            start_cpu_predicition = time.clock()
            example_prediction = self.model.predict(text, self.level)
            end_cpu_predicition = time.clock()
            time_to_delete += (end_cpu_predicition - start_cpu_predicition)
            current_y = numpy.argmax(examples_y[index])
            if current_y == example_prediction:
                adversarial_text, sub_rate, NE_rate, change_tuple_list = self.make_perturbation(text, self.level)
                start_cpu_predicition = time.clock()
                adversarial_prediction = self.model.predict(adversarial_text, self.level)
                end_cpu_predicition = time.clock()
                time_to_delete += (end_cpu_predicition - start_cpu_predicition)
                if adversarial_prediction != current_y:
                    successful_perturbations += 1

                else:
                    failed_perturbations += 1

                text = adversarial_text
                sub_rate_list.append(sub_rate)
                NE_rate_list.append(NE_rate)
                file_changed_words.write(str(index) + str(change_tuple_list) + '\n')

            adversarial_examples.append(text)
            file_adversarial_examples.write(text + "; sub_rate: " + str(sub_rate) + "; NE_rate: " + str(NE_rate) + "\n")

        end_cpu = time.clock()
        mean_sub_rate = sum(sub_rate_list) / len(sub_rate_list)
        mean_NE_rate = sum(NE_rate_list) / len(NE_rate_list)
        support.colored_print("Completed!\nModel: {};\nDataset: {};\nTime elapsed: {} seconds;\nMean substitution rate: {};\nMean NE rate: {}.".format(self.model.name, self.model.phrase_manager.name, (end_cpu - start_cpu - time_to_delete), mean_sub_rate, mean_NE_rate), "blue", verbose)
        file_adversarial_examples.close()
        file_changed_words.close()
        return adversarial_examples

    def make_perturbation(self, text, level):
        pass

    def _read_configuration(self):
        if path.exists(support.get_generator_configuration_path(self.name)):
            with open(support.get_generator_configuration_path(self.name)) as json_file:
                return json.load(json_file)
