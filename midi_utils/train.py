import tensorflow as tf
import keras
import keras_nlp
import json
from .preprocessing import Tokenizer
from .midi_generation import generate_midi

def load_and_prepare_data(training_file, validation_file, tokenizer_file, sequence_length, min_tokens, batch_size):
    
    # import training data and tokenizer fitted on the data
    with open(training_file, 'r') as f:
        training = json.load(f)

    with open(validation_file, 'r') as f:
        validation = json.load(f)

    tokenizer = Tokenizer()
    tokenizer.load(tokenizer_file)

    # pad or truncate to SEQUENCE_LENGTH+1, so we can get input and output sequences of length SEQUENCE_LENGTH shifted by one position
    padded_training = [tokenizer.tokenize(measures, max_length=sequence_length+1) for measures in training]
    padded_validation = [tokenizer.tokenize(measures, max_length=sequence_length+1) for measures in validation]

    training_dataset = (
        tf.data.Dataset.from_tensor_slices(padded_training)
        .filter(lambda x: tf.math.count_nonzero(x) >= min_tokens)
        .shuffle(buffer_size=256)
        .batch(batch_size)
    )

    validation_dataset = (
        tf.data.Dataset.from_tensor_slices(padded_validation)
        .filter(lambda x: tf.math.count_nonzero(x) >= min_tokens)
        .shuffle(buffer_size=256)
        .batch(batch_size)
    )

    # prepare input and output sequences
    def prepare_inputs_outputs(input_batch):
        """
        Shift token sequences by 1 position so that the target for position (i) is word at position (i+1).
        """
        x = input_batch[:, :-1]
        y = input_batch[:, 1:]
        return x, y

    training_dataset_processed = (
        training_dataset.map(prepare_inputs_outputs, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    validation_dataset_processed = (
        validation_dataset.map(prepare_inputs_outputs, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return tokenizer, training_dataset_processed, validation_dataset_processed


def create_model(tokenizer, sequence_length, embedding_dim, feed_forward_dim, num_heads, num_layers, dropout):

    vocab_size = tokenizer.get_vocab_size()

    inputs = keras.layers.Input(shape=(None,), dtype='int32')

    # embedding and positional encoding
    embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=vocab_size,
        sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        mask_zero=True,
    )
    x = embedding_layer(inputs)

    # transformer decoders blocks
    for _ in range(num_layers):
        decoder_layer = keras_nlp.layers.TransformerDecoder(
            num_heads=num_heads,
            intermediate_dim=feed_forward_dim,
            dropout=dropout
        )
        x = decoder_layer(x)  # giving one argument only skips cross-attention.

    # output dense layer
    outputs = keras.layers.Dense(vocab_size)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)
    model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])

    return model


def train_model(output_directory, model, training, validation, epochs, patience=5):

    # EarlyStopping callback
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # CSVLogger callback
    log_path = f'{output_directory}/log.csv'
    csv_logger = keras.callbacks.CSVLogger(log_path, separator=',', append=True)


    # train the model with callbacks
    history = model.fit(
        training,
        validation_data=validation,
        epochs=epochs,
        callbacks=[early_stopping, csv_logger]
    )

    # save the entire model
    model.save(f'{output_directory}/model.keras')

    # save the history to a JSON file
    history_dict = history.history
    with open(f'{output_directory}/training_history.json', 'w') as json_file:
        json.dump(history_dict, json_file)

    return model



def generate_examples(output_directory, model, tokenizer, max_len):

    # generate some examples

    seed_0 = ['MEASURE']

    seed_1 = ['MEASURE', 'EVEN', 'NOTE=43', 'DELTA=2.0', 'NOTE=58', 'DELTA=1.5', 'NOTE=62', 'DELTA=1.5', 'NOTE=67', 'DELTA=1.5', 'STEP', 'DELTA=1.0', 'NOTE=55', 'DELTA=0.5',
            'STEP', 'DELTA=0.5', 'NOTE=57', 'DELTA=0.5', 'NOTE=65', 'DELTA=0.5', 'STEP', 'DELTA=0.25', 'NOTE=63', 'DELTA=0.25', 'STEP', 'DELTA=0.25', 'NOTE=43', 'DELTA=2.0',
            'NOTE=58', 'DELTA=0.5', 'NOTE=62', 'DELTA=0.5', 'STEP', 'DELTA=0.5', 'NOTE=60', 'DELTA=0.5', 'STEP', 'DELTA=0.5', 'NOTE=62', 'DELTA=1.0', 'NOTE=70', 'DELTA=1.0',
            'STEP', 'DELTA=1.0', 'MEASURE', 'EVEN', 'NOTE=51', 'DELTA=2.0', 'NOTE=55', 'DELTA=2.0', 'NOTE=63', 'DELTA=1.0', 'STEP', 'DELTA=0.5', 'NOTE=72', 'DELTA=0.25',
            'STEP', 'DELTA=0.25', 'NOTE=70', 'DELTA=0.25', 'STEP', 'DELTA=0.25', 'NOTE=69', 'DELTA=0.5', 'STEP', 'DELTA=0.5', 'NOTE=67', 'DELTA=0.5']

    results = []

    for seed_index, seed in enumerate([seed_0, seed_1]):
        for proba in [0.6, 0.7, 0.8]:
            for index in range(2):
                generated_sequence, midi_object = generate_midi(
                    model=model,
                    tokenizer=tokenizer,
                    seed=seed,
                    max_len=max_len,
                    p=proba,
                    tempo=100,
                    save_midi=True,
                    output_filepath=f'{output_directory}/test_runs/output_seed{seed_index+1}_prob{proba}_{index+1}.mid',
                )
                results.append((generated_sequence, midi_object))

    return results
