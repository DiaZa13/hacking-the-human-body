from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate

class UNET:
    def __init__(self, img_size, neurons, channels,  factors, dropout):
        self.size = img_size
        self.neurons = neurons
        self.factors = factors
        self.dropout = dropout
        self.input = Input((self.size, self.size, channels))

    def _convolution(self, factor: int, dropout: float, last_layer):
        c = Conv2D(self.neurons * factor, (3, 3), activation='relu', padding='same')(last_layer)
        c = Dropout(dropout)(c)
        return Conv2D(self.neurons * factor, (3, 3), activation="relu", padding="same")(c)

    def _encoder_base(self, factor: int, dropout: float, last_layer):
        c = self._convolution(factor, dropout, last_layer)
        p = MaxPooling2D((2, 2))(c)
        return c, p

    def _decoder_base(self, factor: int, dropout: float, last_layer, concatenate_layer):
        dc = Conv2DTranspose(self.neurons * factor, (3, 3), strides=(2, 2), padding="same")(last_layer)
        dc = Concatenate()([dc, concatenate_layer])
        uc = self._convolution(factor, dropout, dc)
        return uc

    def _build_model(self):
        c1, p1 = self._encoder_base(self.factors[0], self.dropout, self.input)
        c2, p2 = self._encoder_base(self.factors[1], self.dropout, p1)
        c3, p3 = self._encoder_base(self.factors[2], self.dropout, p2)
        c4, p4 = self._encoder_base(self.factors[3], self.dropout, p3)

        # middle
        b1 = self._convolution(self.factors[4], self.dropout, p4)

        d1 = self._decoder_base(self.factors[3], self.dropout, b1, c4)
        d2 = self._decoder_base(self.factors[2], self.dropout, d1, c3)
        d3 = self._decoder_base(self.factors[1], self.dropout, d2, c2)
        d4 = self._decoder_base(self.factors[0], self.dropout, d3, c1)

        return Conv2D(1, 1, padding='same', activation="sigmoid")(d4)

    def model(self, optimizer, loss, metrics):
        output = self._build_model()
        umodel = Model(self.input, output)
        umodel.compile(optimizer, loss, metrics)
        return umodel