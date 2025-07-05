# model_definitions.py - Updated with SMILElength and CNNaffinity conditional features
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal  # Gupta Used this
from config import CONFIG
from tensorflow.keras import regularizers
# Handle different TensorFlow versions for serialization decorator
try:
    # TensorFlow 2.9+
    from tensorflow.keras.saving import register_keras_serializable

    keras_serializable = register_keras_serializable()

except (ImportError, AttributeError):
    try:
        # TensorFlow 2.7-2.8
        from tensorflow.keras.utils import register_keras_serializable

        keras_serializable = register_keras_serializable()
    except (ImportError, AttributeError):
        # Older TensorFlow versions - use custom serializable decorator
        def keras_serializable(cls):
            return cls



@keras_serializable
class Attention(layers.Layer):
    """
    Standard Additive Attention layer (Bahdanau-style).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # --- FIX: Define layers in __init__ ---
        # The number of units can be arbitrary, but using the values_dim is standard.
        # We will define the units in the build method once we know the input shape.
        self.W1 = None
        self.W2 = None
        self.V = None

    def build(self, input_shape):
        # input_shape is a list of shapes [query_shape, values_shape]
        values_dim = input_shape[-1]

        # --- CORRECTED: Use Dense layers to handle weights and matrix multiplication ---
        self.W1 = layers.Dense(values_dim, name="att_dense_1")
        self.W2 = layers.Dense(values_dim, name="att_dense_2")
        self.V = layers.Dense(1, name="att_dense_v")

        super().build(input_shape)

    def call(self, query, values):
        """
        Calculates Bahdanau-style attention.
        Args:
            query: The decoder hidden state. Shape: (batch_size, rnn_units).
            values: The encoder/decoder outputs. Shape: (batch_size, seq_len, rnn_units).
        Returns:
            A tuple of (context_vector, attention_weights).
        """
        query_with_time_axis = tf.expand_dims(query, 1)

        # --- CORRECTED: The Dense layers are now callable ---
        # The layers apply the dot product and add a bias (by default).
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)
        ))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights




@keras_serializable
class ConditionalRNNGenerator(tf.keras.Model):
    def __init__(self, vocab_size,
                 embedding_dim,
                 rnn_units,  # rnn_units should be 1024 for the 3 LSTMs
                 sa_embedding_dim=0,
                 slogp_embedding_dim=0,
                 exactmw_embedding_dim=0,
                 numrotatablebonds_embedding_dim=0,
                 cnnscore_embedding_dim=0,
                 CNNaffinity_embedding_dim=0,
                 CNNminAffinity_embedding_dim=0,
                 SMILElength_embedding_dim=0,  # ADDED
                 numaromaticrings_embedding_dim=0,
                 numhbd_embedding_dim=0,
                 numhba_embedding_dim=0,
                 dropout_rate=0.2,  # Added dropout_rate parameter
                 random_seed=42,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units  # This will be 1024 for each of the 3 LSTMs
        self.dropout_rate = dropout_rate
        self.random_seed = random_seed
        self.sa_embedding_dim = sa_embedding_dim
        self.slogp_embedding_dim = slogp_embedding_dim
        self.exactmw_embedding_dim = exactmw_embedding_dim
        self.numrotatablebonds_embedding_dim = numrotatablebonds_embedding_dim
        self.cnnscore_embedding_dim = cnnscore_embedding_dim
        self.CNNaffinity_embedding_dim = CNNaffinity_embedding_dim
        self.CNNminAffinity_embedding_dim = CNNminAffinity_embedding_dim
        self.smilelength_embedding_dim = SMILElength_embedding_dim  # ADDED
        self.numaromaticrings_embedding_dim = numaromaticrings_embedding_dim
        self.numhbd_embedding_dim = numhbd_embedding_dim
        self.numhba_embedding_dim = numhba_embedding_dim

        if self.embedding_dim > 0:
            self.embedding = layers.Embedding(vocab_size,
                                              embedding_dim,
                                              name="token_embedding")
        else:
            self.embedding = None  # Explicitly set to None if embedding_dim is 0


        # Conditional Feature Processors
        if self.sa_embedding_dim > 0:
            self.sa_dense_processor = layers.Dense(sa_embedding_dim, activation='relu',
                                                   name='sa_processor')
        else:
            self.sa_dense_processor = None

        if self.slogp_embedding_dim > 0:
            self.slogp_dense_processor = layers.Dense(slogp_embedding_dim,
                                                      activation='relu',
                                                      name='slogp_processor')
        else:
            self.slogp_dense_processor = None

        if self.exactmw_embedding_dim > 0:
            self.exactmw_dense_processor = layers.Dense(exactmw_embedding_dim,
                                                        activation='relu',
                                                        name='exactmw_processor')
        else:
            self.exactmw_dense_processor = None

        if self.numrotatablebonds_embedding_dim > 0:
            self.numrotatablebonds_dense_processor = layers.Dense(numrotatablebonds_embedding_dim,
                                                                  activation='relu',
                                                                  name='numrotatablebonds_processor')
        else:
            self.numrotatablebonds_dense_processor = None

        if self.cnnscore_embedding_dim > 0:
            self.cnnscore_dense_processor = layers.Dense(cnnscore_embedding_dim,
                                                         activation='relu',
                                                         name='cnnscore_processor')
        else:
            self.cnnscore_dense_processor = None

        if self.CNNaffinity_embedding_dim > 0:
            self.CNNaffinity_dense_processor = layers.Dense(CNNaffinity_embedding_dim,
                                                            activation='relu',
                                                            name='cnnaffinity_processor')
        else:
            self.CNNaffinity_dense_processor = None

        if self.CNNminAffinity_embedding_dim > 0:
            self.CNNminAffinity_dense_processor = layers.Dense(CNNminAffinity_embedding_dim,
                                                               activation='relu',
                                                               name='CNNminAffinity_processor')
        else:
            self.CNNminAffinity_dense_processor = None

        # --- ADDED SMILELENGTH PROCESSOR ---
        if self.smilelength_embedding_dim > 0:
            self.SMILElength_dense_processor = layers.Dense(SMILElength_embedding_dim,
                                                            activation='relu',
                                                            name='smilelength_processor')
        else:
            self.SMILElength_dense_processor = None
        # --- END OF ADDITION ---

        if self.numaromaticrings_embedding_dim > 0:
            self.numAromaticRings_dense_processor = layers.Dense(numaromaticrings_embedding_dim,
                                                                 activation='relu',
                                                                 name='numaromaticrings_processor')
        else:
            self.numAromaticRings_dense_processor = None

        if self.numhbd_embedding_dim > 0:
            self.numhbd_dense_processor = layers.Dense(numhbd_embedding_dim,
                                                       activation='relu',
                                                       name='numhbd_processor')
        else:
            self.numhbd_dense_processor = None

        if self.numhba_embedding_dim > 0:
            self.numhba_dense_processor = layers.Dense(numhba_embedding_dim,
                                                       activation='relu',
                                                       name='numhba_processor')
        else:
            self.numhba_dense_processor = None

        weight_init = RandomNormal(mean=0.0, stddev=0.05, seed=self.random_seed)


        # Begin Gupta LSTM model and Segler's model
        self.lstm1 = layers.LSTM(rnn_units,
                                 return_sequences=True,
                                 kernel_initializer=weight_init,
                                 return_state=True,
                                 #recurrent_initializer='glorot_uniform',
                                 name="lstm_layer_1",
                                 activation='tanh',
                                 dropout=self.dropout_rate)

        # Bojum
        if not CONFIG['Bjerrum'] and not CONFIG['Gupta']:
            self.dropout1 = layers.Dropout(self.dropout_rate, name="dropout_layer_1")
            self.lstm2 = layers.LSTM(rnn_units,
                                     return_sequences=True, #True
                                     kernel_initializer=weight_init,
                                     return_state=True,
                                     #recurrent_initializer='glorot_uniform',
                                     name="lstm_layer_2",
                                     activation='tanh'
                                     )
            self.dropout2 = layers.Dropout(self.dropout_rate, name="dropout_layer_2")

        self.lstm3 = layers.LSTM(rnn_units,
                                 return_sequences=True,
                                 return_state=True,
                                 kernel_initializer=weight_init,
                                 #recurrent_initializer='glorot_uniform',
                                 name="lstm_layer_3",
                                 activation='tanh',
                                 dropout=self.dropout_rate)
        #
        if not CONFIG['Bjerrum'] and not CONFIG['Gupta']:
            self.dropout3 = layers.Dropout(self.dropout_rate, name="dropout_layer_3")

        # --- FEED FORWARD NETWORK Like Bjerrum ---
        if CONFIG['Bjerrum'] and not CONFIG['Gupta']:
            self.ff_dense1 = layers.TimeDistributed(layers.Dense(128, activation='relu', name="ff_dense_1"))
            self.ff_dense2 = layers.TimeDistributed(layers.Dense(128, activation='relu', name="ff_dense_2"))

        # --- ATTENTION LAYER ---
        if CONFIG['attention']:
            self.attention = Attention(name="attention_layer")
        # --- END OF ADDITION ---
        if CONFIG['Bjerrum']:
            self.dense = layers.Dense(vocab_size,
                                      name="output_dense",
                                      # Shown in Bejrum's online code, but not mentioned in paper
                                      kernel_regularizer=regularizers.l1_l2(0.005, 0.01)
                                      )
        else:
            self.dense = layers.Dense(vocab_size, name="output_dense")


    def build(self, input_shape):
        """Build the model layers."""
        # Call build on embedding layer if it exists
        if self.embedding is not None:
            self.embedding.build((None, None))  # (batch_size, sequence_length)

        # Build all the dense processors if they exist
        processors = [
            self.sa_dense_processor,
            self.slogp_dense_processor,
            self.exactmw_dense_processor,
            self.numrotatablebonds_dense_processor,
            self.cnnscore_dense_processor,
            self.CNNaffinity_dense_processor,
            self.CNNminAffinity_dense_processor,
            self.SMILElength_dense_processor,
            self.numAromaticRings_dense_processor,
            self.numhbd_dense_processor,
            self.numhba_dense_processor
        ]

        for processor in processors:
            if processor is not None:
                processor.build((None, 1))  # (batch_size, 1) for auxiliary features

        # The LSTM and Dense layers will be built automatically when first called
        super().build(input_shape)

    def call(self, inputs, states=None, training=False):
        # Determine if inputs are a dictionary (indicating auxiliary features are present)
        is_dict_input = isinstance(inputs, dict)

        if is_dict_input:
            token_input_seq = inputs['tokens']
            # Only include auxiliary features if their embedding dimension is > 0 and they are in inputs
            auxiliary_features_to_process = []
            if self.cnnscore_embedding_dim > 0 and 'cnn' in inputs:
                auxiliary_features_to_process.append(('cnn', self.cnnscore_embedding_dim, self.cnnscore_dense_processor, inputs['cnn']))
            if self.CNNaffinity_embedding_dim > 0 and 'cnnAff' in inputs:
                auxiliary_features_to_process.append(('cnnAff', self.CNNaffinity_embedding_dim, self.CNNaffinity_dense_processor, inputs['cnnAff']))
            if self.smilelength_embedding_dim > 0 and 'smilen' in inputs:
                auxiliary_features_to_process.append(('smilen', self.smilelength_embedding_dim, self.SMILElength_dense_processor, inputs['smilen']))
            if self.CNNminAffinity_embedding_dim > 0 and 'CNNminAff' in inputs:
                auxiliary_features_to_process.append(('CNNminAff', self.CNNminAffinity_embedding_dim, self.CNNminAffinity_dense_processor, inputs['CNNminAff']))
        else:
            token_input_seq = inputs
            auxiliary_features_to_process = [] # No auxiliary features if not dict input

        if self.embedding_dim > 0:
            token_embedded = self.embedding(token_input_seq)
        else:
            # If embedding_dim is 0, the data preprocessor should have already
            # one-hot encoded the tokens. So, token_input_seq IS the token_embedded.
            # We just need to ensure its dtype is float32 for consistency with auxiliary_embeddings.
            token_embedded = tf.cast(token_input_seq, tf.float32)

        auxiliary_embeddings = []
        for name, feature_dim, processor, feature_value in auxiliary_features_to_process:
            if processor is not None: # Ensure processor exists
                if len(feature_value.shape) == 1:
                    feature_value = tf.expand_dims(feature_value, -1)
                feature_embedded = processor(feature_value)
                # Tile the embedded feature across the sequence length dimension
                feature_tiled = tf.tile(tf.expand_dims(feature_embedded, axis=1), [1, tf.shape(token_embedded)[1], 1])
                auxiliary_embeddings.append(feature_tiled)


        if auxiliary_embeddings:
            combined_input = tf.concat([token_embedded] + auxiliary_embeddings, axis=-1)
        else:
            combined_input = token_embedded

        # STEP 5: Pass through LSTM layers, managing states correctly where present
        x = combined_input

        s1 = states[0:2] if states is not None and len(states) >= 2 else None

        if not CONFIG['Bjerrum'] and not CONFIG['Gupta']:
            s2 = states[2:4] if states is not None and len(states) >= 4 else None

        s3 = states[4:6] if states is not None and len(states) >= 6 else None
        # If Bjerrum or Gupta is True, then s3 should be states[2:4] because there are only two LSTMs
        if (CONFIG['Bjerrum'] or CONFIG['Gupta']) and (states is not None and len(states) >= 4):
            s3 = states[2:4]

        x, h1, c1 = self.lstm1(x, initial_state=s1, training=training)

        if not CONFIG['Bjerrum'] and not CONFIG['Gupta']:
            x = self.dropout1(x, training=training)
            x, h2, c2 = self.lstm2(x, initial_state=s2, training=training)
            x = self.dropout2(x, training=training)


        lstm3_output, h3, c3 = self.lstm3(x, initial_state=s3, training=training)

        if CONFIG['Bjerrum'] and not CONFIG['Gupta']:
            x = self.ff_dense1(lstm3_output)
            z = self.ff_dense2(x)
        else:
            if CONFIG['attention']:
                context_vector, attention_weights = self.attention(h3, lstm3_output)
                context_vector_expanded = tf.expand_dims(context_vector, 1)
                z = tf.nn.tanh(lstm3_output + context_vector_expanded)
            else:
                z = lstm3_output


        output = self.dense(z)

        if training:
            return output
        else:
            if not CONFIG['Bjerrum'] and not CONFIG['Gupta']:
                return output, [h1, c1, h2, c2, h3, c3]
            else:
                return output, [h1, c1, h3, c3]


    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "rnn_units": self.rnn_units,
            "dropout_rate": self.dropout_rate,
            "sa_embedding_dim": self.sa_embedding_dim,
            "slogp_embedding_dim": self.slogp_embedding_dim,
            "exactmw_embedding_dim": self.exactmw_embedding_dim,
            "numrotatablebonds_embedding_dim": self.numrotatablebonds_embedding_dim,
            "cnnscore_embedding_dim": self.cnnscore_embedding_dim,
            "CNNaffinity_embedding_dim": self.CNNaffinity_embedding_dim,
            "CNNminAffinity_embedding_dim": self.CNNminAffinity_embedding_dim,
            "SMILElength_embedding_dim": self.smilelength_embedding_dim,  # ADDED
            "numaromaticrings_embedding_dim": self.numaromaticrings_embedding_dim,
            "numhbd_embedding_dim": self.numhbd_embedding_dim,
            "numhba_embedding_dim": self.numhba_embedding_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        if 'dropout_rate' not in config:
            config['dropout_rate'] = 0.2
        # --- ADDED: Default for SMILElength_embedding_dim for backward compatibility ---
        if 'SMILElength_embedding_dim' not in config:
            config['SMILElength_embedding_dim'] = 0
        # --- END OF ADDITION ---
        return cls(**config)