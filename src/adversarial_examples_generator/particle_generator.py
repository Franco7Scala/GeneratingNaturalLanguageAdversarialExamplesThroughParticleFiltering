import numpy
import spacy

from src.adversarial_examples_generator.adversarial_examples_generator import AdversarialExampleGenerator
from src.support import support


QUANTITY_PARTICLES = 0 #TODO place in a right location
STEPS = 0 #TODO place in a right location


class ParticleGenerator(AdversarialExampleGenerator):

    def __init__(self, model, level):
        super().__init__(model, level)
        self.nlp = spacy.load('en', tagger=False, entity=False)
        self.change_tuple_list = []

    def make_perturbation(self, text, level):
        self.change_tuple_list = []
        doc = self.nlp(text)
        particles = numpy.full(len(self.configuration[QUANTITY_PARTICLES]), doc)
        distances = numpy.full(len(particles), numpy.inf)
        for step in range(0, STEPS):
            particles, substitutions, distances = self._move_particles(particles, substitutions, distances)
            # Movimento: funzioni da provare
                # Precaricarsi per ogni posizione le possibili sostituzioni (top k più vicene rispetto a glove/word2vec)
                # Per ogni parola tirare un dado p in [0,1] scegliere:
                # Se p < pself lasci la parola che c'è nella particella
                # Altrimenti:
                        # 1 alternativa - movimento casuale) scegli una parola a caso
                        # 2 alternativa - movimento per similarità) scegli la parola da sostituire sulla base dell'inverso della distanza dalla parola originale (righiede il calcolo delle distanze all'inizio)
            # Se non arresto
                #Filtro partieelle attraverso campionamento generando QUANTITY_PARTICLES campioni(particelle):
                    # 1 alternativa) campiono le particelle assegnandogli come probabilità il guadagno di classificazione inteso come lo score della classificazione della particella per la/e classi target
                    # 2 alternativa) campiono le particelle assegnandogli come probabilità il guadagno di classificazione/distanza dalla particella originaria

        return self._select_particle(), self.change_tuple_list

    def _move_particles(self, texts, substitutions, distances):
        pass #returns texts, substitutions, distances

    def _calculate_distances(self):
        pass

    def _select_particle(self):
        pass #returns text selected, (substitute_count / len(doc)), NE_rate = NE_count / substitute_count
