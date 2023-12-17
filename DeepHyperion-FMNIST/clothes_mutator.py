import random
import mutation_manager
import rasterization_tools
import vectorization_tools
from properties import MUTOPPROB, MUTOFPROB, DISTANCE, TSHD_TYPE, DISTANCE_SEED
from utils import get_distance, reshape
#from tensorflow import keras
import keras


class ClothesMutator:
    # load the FMNIST dataset
    fmnist = keras.datasets.fashion_mnist
    (_, _), (x_test, _) = fmnist.load_data()

    def __init__(self, clothes):
        self.clothes = clothes

    def mutate(self):
        condition = True
        counter_mutations = 0
        while condition:
            # Select mutation operator.
            rand_mutation_probability = random.uniform(0, 1)
            rand_mutation_prob = random.uniform(0, 1)
            if rand_mutation_probability >= MUTOPPROB:            
                if rand_mutation_prob >= MUTOFPROB:
                    mutation = 1
                else:
                    mutation = 2
            else:
                if rand_mutation_prob >= MUTOFPROB:
                    mutation = 3
                else:
                    mutation = 4

            counter_mutations += 1
            mutant_vector = mutation_manager.mutate(self.clothes.xml_desc, mutation, counter_mutations/20)
            mutant_xml_desc = vectorization_tools.create_svg_xml(mutant_vector)
            rasterized_clothes = rasterization_tools.rasterize_in_memory(mutant_xml_desc)

            distance_inputs = get_distance(self.clothes.purified, rasterized_clothes)

            if (TSHD_TYPE == '0'):
                if distance_inputs != 0:
                    condition = False
            elif (TSHD_TYPE == '1'):
                seed_image = ClothesMutator.x_test[int(self.clothes.seed)]
                xml_desc = vectorization_tools.vectorize(seed_image)
                seed = rasterization_tools.rasterize_in_memory(xml_desc)
                distance_seed = get_distance(seed, rasterized_clothes)
                if distance_inputs != 0 and distance_seed <= DISTANCE and distance_seed != 0:
                    condition = False
            elif (TSHD_TYPE == '2'):
                seed = reshape(ClothesMutator.x_test[int(self.clothes.seed)])
                distance_seed = get_distance(seed, rasterized_clothes)
                if distance_inputs != 0 and distance_seed <= DISTANCE_SEED and distance_seed != 0:
                    condition = False

        self.clothes.xml_desc = mutant_xml_desc
        self.clothes.purified = rasterized_clothes
        self.clothes.predicted_label = None
        self.clothes.confidence = None

