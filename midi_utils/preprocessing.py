from sklearn.model_selection import train_test_split

import os
import json


class Tokenizer:
    """
    A simple tokenizer class for converting sequences of tokens to indices and vice versa.

    Attributes:
    - vocab (dict): A dictionary mapping tokens to their corresponding indices.
    - reverse_vocab (dict): A dictionary mapping indices to their corresponding tokens.

    Methods:
    - fit(input_data): Build the vocabulary based on the input sequence (string or list).
    - tokenize(input_data, max_length=None): Convert a sequence of tokens (string or list) to a list of indices.
        If max_length is provided, sequences are truncated or padded on the right.
    - detokenize(token_sequence): Convert a sequence of indices to a list of tokens.
    - get_vocab_size(): Get the size of the vocabulary.
    - save(json_filepath): saves the tokenizer to filepath.
    - load(json_filepath): loads a

    Note:
    - Special tokens '[PAD]', '[UNKNOWN]', '[START]', and '[END]' are automatically added to the vocabulary.
    - Tokens not present in the vocabulary are mapped to '[UNKNOWN]'.
    - The vocabulary is built using the `fit` method before using `tokenize` or `detokenize`.
    """
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}

    def fit(self, input_data):
        if isinstance(input_data, str):
            input_sequence = input_data.split()
        elif isinstance(input_data, list):
            input_sequence = input_data
        else:
            raise ValueError("Input must be either a string or a list of elements.")

        token_counts = {}
        for token in input_sequence:
            token_counts[token] = token_counts.get(token, 0) + 1

        sorted_tokens = sorted(token_counts.keys(), key=lambda x: token_counts[x], reverse=True)
        sorted_tokens = ['[PAD]', '[UNKNOWN]', '[START]', '[END]'] + sorted_tokens

        self.vocab = {token: idx for idx, token in enumerate(sorted_tokens)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def tokenize(self, input_data, max_length=None):
        if isinstance(input_data, str):
            input_sequence = input_data.split()
        elif isinstance(input_data, list):
            input_sequence = input_data
        else:
            raise ValueError("Input must be either a string or a list of elements.")
        tokens = [self.vocab.get(token, self.vocab['[UNKNOWN]']) for token in input_sequence]

        # truncating and padding (right)
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]

        if max_length is not None and len(tokens) < max_length:
            tokens += [self.vocab['[PAD]']] * (max_length - len(tokens))

        return tokens

    def detokenize(self, token_sequence):
        return [self.reverse_vocab.get(token, '[UNKNOWN]') for token in token_sequence]

    def get_vocab_size(self):
        return len(self.vocab)

    def save(self, json_filepath):
        data = {'vocab': self.vocab, 'reverse_vocab': self.reverse_vocab}
        with open(json_filepath, 'w') as f:
            json.dump(data, f)

    def load(self, json_filepath):
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.reverse_vocab = {int(k): v for k, v in data['reverse_vocab'].items()}
    

def split_into_measures(input_list, n):
    """
    Split a sequence of musical events into subsequences, each containing a maximum of n measures.
    
    Args:
    - input_list (list): A list of musical events or a list of such lists.
    - n (int): Number of measures to use for splitting.
    
    Returns:
    - output_list (list): A list of subsequences, each containing a maximum of n measures.
    
    Note:
    - The function identifies measures based on the occurrence of 'MEASURE' events.
    """
    def split_into_sublists(token_sequence, n=n):
        measure_indices = [i for i, token in enumerate(token_sequence) if token == 'MEASURE'][::n][1:]
        sublists = []
        start = 0
        for index in measure_indices:
            sublists.append(token_sequence[start:index])
            start = index
            
        # Add the last sublist from the last index to the end of the list
        sublists.append(token_sequence[start:])
        return sublists

    if isinstance(input_list[0], str):
        return split_into_sublists(input_list, n=n)
        
    if isinstance(input_list[0], list):
        output_list = []
        for token_sequence in input_list:
            output_list.extend(split_into_sublists(token_sequence, n=n))
            
        return output_list


def split_and_transpose(parsed_midis, df, n_bars, transpose=False, save_data=False, output_directory=None):
    """
    Split and optionally transpose MIDI data into measures for training a music generation model.

    Args:
    - parsed_midis (dict): A dictionary containing parsed MIDI data, where keys are file names
    and values are dictionaries with 'events' representing the musical events in MIDI format.
    - df (pandas.dataframe): Dataframe with a Key column containing the keys of the pieces in midi_data.
    - n_bars (int): Number of bars per measure for splitting the data.
    - transpose (bool): If True, transpose each measure to multiple keys for data augmentation.
    - save_data (bool): If True, save the training and validation data, and the tokenizer.
    - output_directory (str): Path to the directory where data and tokenizer are saved if save_data is True.

    Returns:
    - measures (list): A list of measures, each represented as a list of musical events.
    - training (list): A list of training measures for model training.
    - validation (list): A list of validation measures for model evaluation.
    - tokenizer (Tokenizer): A Tokenizer instance for converting tokens to indices.

    Note:
    - The function splits MIDI data into measures and optionally transposes them for data augmentation.
    - Training and validation sets are created using a 90-10 split.
    - If save_data is True, training and validation data are saved to 'training_data.txt' and 'validation_data.txt',
    and the Tokenizer is saved to 'tokenizer.pkl'.
    """
    def transpose(measure, n):
        transposed_measure = []
        for event in measure:
            if event.startswith('NOTE'):
                _, pitch = event.split('=')
                transposed_measure.append(f'NOTE={int(pitch)+n}')
            else:
                transposed_measure.append(event)  
        return transposed_measure
    
    pieces = []
    for file_name, data in parsed_midis.items():
        index = file_name.strip('file_')
        index = index.strip('.mid')
        index = int(index)
        key = df.loc[index].Key
        event_data = data['events']
        pieces.append(event_data)
    
        # transpose two half steps up or down
        if transpose:
            if key.startswith('C'):
                transposed = transpose(event_data, 2) # C to D
                pieces.append(transposed)
            if key.startswith('D'):
                transposed = transpose(event_data, -2) # D to C
                pieces.append(transposed)
            if key.startswith('F'):
                transposed = transpose(event_data, 2) # F to G
                pieces.append(transposed)
            if key.startswith('G'):
                transposed = transpose(event_data, -2) # G to F
                pieces.append(transposed)
    
    measures = split_into_measures(pieces, n=n_bars)
    tokenizer = Tokenizer()
    measures_flat = [event for measure in measures for event in measure]
    tokenizer.fit(measures_flat)

    VALIDATION_SIZE = 0.1
    training, validation = train_test_split(measures, test_size=VALIDATION_SIZE, random_state=4)

    if save_data:
        
        with open(os.path.join(output_directory, 'training_data.json'), 'w') as f:
            json.dump(training, f)
                
        with open(os.path.join(output_directory, 'validation_data.json'), 'w') as f:
            json.dump(validation, f)

        tokenizer.save(os.path.join(output_directory, 'tokenizer.json'))

    return measures, training, validation, tokenizer