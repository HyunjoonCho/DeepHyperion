import predictor
from digit_mutator import DigitMutator


class Individual(object):
    # Global counter of all the individuals (it is increased each time an individual is created or mutated).
    COUNT = 0
    SEEDS = set()
    COUNT_MISS = 0

    def __init__(self, member, seed):
        self.seed = seed
        self.ff = None
        self.member = member

    def reset(self):
        self.ff = None

    def evaluate(self):
        if self.ff is None:          
            self.member.predicted_label1, self.member.predicted_label2, self.member.activation_difference = \
                predictor.Predictor.predict(self.member.purified)

            # Calculate fitness function 
            # TODO: need tuning of fitness cacluation / may flip the minimization flag
            self.ff = self.member.activation_difference / 5 if self.member.predicted_label1 == self.member.predicted_label2 else -0.1
            
        return self.ff

    def mutate(self):
        DigitMutator(self.member).mutate()
        self.reset()

