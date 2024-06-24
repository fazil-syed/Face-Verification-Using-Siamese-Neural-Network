# Custom L1 Distance layer module 
# WHY DO WE NEED THIS: its needed to load the custom model

# Import dependencies
import tensorflow as tf


# Custom L1 Distance Layer from Jupyter 
class L1Dist(tf.keras.layers.Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        input_embedding = tf.convert_to_tensor(input_embedding)
        validation_embedding = tf.convert_to_tensor(validation_embedding)
        return tf.math.abs(input_embedding - validation_embedding)