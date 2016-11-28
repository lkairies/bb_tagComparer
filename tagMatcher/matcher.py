from diktya.func_api_helpers import load_model
import numpy as np


class Matcher():
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.model._make_predict_function()

    @staticmethod
    def _int_to_bit_array(n, length):
        bits = [1 if digit == '1' else -1 for digit in bin(n)[2:]]
        if len(bits) < length:
            bits = [0]*(length-len(bits)) + bits
        return bits

    # match two tags
    def match(self, representation_a, representation_b):
        if isinstance(representation_a, int):
            # int128 representation
            representation_a = np.array([
                Matcher._int_to_bit_array(representation_a, 128)])
            representation_b = np.array([
                Matcher._int_to_bit_array(representation_b, 128)])
        elif len(representation_a) == 16:
            # representations from bb_binary format (16 uint8 values)
            representation_a = np.array([
                Matcher._int_to_bit_array(rep, 8)
                for rep in representation_a]).flatten()
            representation_b = np.array([
                Matcher._int_to_bit_array(rep, 8)
                for rep in representation_b]).flatten()
        else:
            # 128 float32 representation
            representation_a = np.array([representation_a])
            representation_b = np.array([representation_b])
        return self.model.predict([representation_a.reshape((1, 128)),
                                   representation_b.reshape((1,128))])[0, 0]

    def matchMany(self, representation_a, representations):
        if isinstance(representation_a, int):
            # int128 representation
            representations_a = np.array([Matcher._int_to_bit_array(
                representation_a, 128)] * len(representations))
            representations_b = np.array([Matcher._int_to_bit_array(rep, 128)
                                          for rep in representations])
        elif len(representation_a) == 16:
            # representation from bb_binary format (16 unit8 values)
            representations_a = np.array([Matcher._int_to_bit_array(rep, 8)
                                          for rep in representation_a]
                                         * len(representations))
            representations_a = representations_a.reshape(
                (len(representations), 128))
            representations_b = []
            for representation in representations:
                representations_b.append(np.array([
                    Matcher._int_to_bit_array(rep, 8)
                    for rep in representation]))
            representations_b = np.array(representations_b).reshape(
                (len(representations), 128))
        else:
            # 128 float32 representation
            representations_a = np.array([representation_a] *
                                         len(representations))
            representations_b = representations
        return self.model.predict(
            [representations_a, representations_b]).flatten()
