from diktya.func_api_helpers import load_model
import numpy as np


class Matcher():
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.model._make_predict_function()

    @staticmethod
    def _int_to_bit_array(n, length):
        bits = [1 if digit == '1' else 0 for digit in bin(n)[2:]]
        if len(bits) < length:
            bits = [0]*(length-len(bits)) + bits
        return bits

    @staticmethod
    def _representation_to_bit_representation(representation):
        if isinstance(representation, int):
            # int128 representation
            representation = np.array(
                Matcher._int_to_bit_array(representation, 128))
        elif len(representation) == 16:
            # representations from bb_binary format (16 uint8 values)
            representation = np.array([
                Matcher._int_to_bit_array(n, 8)
                for n in representation]).flatten()
        return representation

    # match two tags
    def match(self, representation_a, representation_b):
        representation_a = Matcher._representation_to_bit_representation(representation_a)
        representation_b = Matcher._representation_to_bit_representation(representation_b)
        return self.model.predict([representation_a.reshape((1, 128)),
                                   representation_b.reshape((1, 128))])[0, 0]

    # match tags pairwise
    def matchPairs(self, representations_a, representations_b):
        assert len(representations_a) == len(representations_b), 'inputs do not have ' \
            'the same length'
        representations_a = np.array([Matcher._representation_to_bit_representation(representation)
                                      for representation in representations_a])
        representations_b = np.array([Matcher._representation_to_bit_representation(representation)
                                      for representation in representations_b])
        return self.model.predict([representations_a, representations_b]).flatten()

    # match one tag with multiple tags
    def matchMany(self, representation_a, representations_b):
        representations_a = np.tile(representation_a, len(representations_b))
        if isinstance(representation_a, int):
            representation_a.reshape((len(representations_b, 1)))
        else:
            representations_a = representations_a.reshape((len(representations_b),
                                                           len(representation_a)))
        return self.matchPairs(representations_a, representations_b)
