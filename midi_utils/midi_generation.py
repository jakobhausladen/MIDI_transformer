import numpy as np

from .encoding_decoding import get_events, decode_tokens_to_midi


##### SAMPLING #####

def sample_top_p(logits, p=0.8):

    sorted_logits = np.sort(logits)[::-1]  # sort in descending order
    sorted_indices = np.argsort(logits)[::-1]  # corresponding indices

    cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits)))

    # find the smallest index whose cumulative probability exceeds p
    sampled_index = np.min(np.where(cumulative_probs > p))

    # use the indices to get the sampled token index
    sampled_token_index = sorted_indices[:sampled_index + 1]

    return np.random.choice(sampled_token_index)


##### SYNTAX CHECKING #####

def fits_previous_token(sampled_event, previous_events):
    """
    Takes a sampled event and a list of events that come before it, and checks if the sampled event is allowed given the final event of the previous ones.
    For instance, <STEP, DELTA> is valid, <STEP, STEP> is not.
    """
    sampled_event_initial = sampled_event[0]

    # padding can be predicted after any token and stops the generation process
    if sampled_event == '[PAD]':
        return True
    else:
        previous_event = previous_events[-1]
        previous_event_initial = previous_event[0]

        # compatibility matrix (current, previous); NOTE, STEP, DELTA, MEASURE, EVEN, TRIPLE
        event_combinations = {
            'N':{'N':False, 'S':False, 'D':True, 'M':False, 'E':True, 'T':True},
            'S':{'N':False, 'S':False, 'D':True, 'M':False, 'E':True, 'T':True}, # we could require note onsets after a new measure
            'D':{'N':True, 'S':True, 'D':False, 'M':False, 'E':False, 'T':False},
            'M':{'N':False, 'S':False, 'D':True, 'M':False, 'E':False, 'T':False},
            'E':{'N':False, 'S':False, 'D':False, 'M':True, 'E':False, 'T':False},
            'T':{'N':False, 'S':False, 'D':False, 'M':True, 'E':False, 'T':False}
        }

        fits_previous = event_combinations[sampled_event_initial][previous_event_initial]
        if not fits_previous:
            print('Incompatible with previous token: resample')

        return fits_previous


def no_note_duplication(sampled_event, previous_events):
    """
    Takes a sampled event and a list of events that come before it, and checks if appending the former to the latter results in a NOTE duplication,
    i.e. if the same NOTE event occurs twice between two STEP events.
    """
    if not sampled_event.startswith('NOTE'):
        return True

    # find the index of the last STEP event
    last_step_index = -1
    for i, event in enumerate(previous_events):
        if event == 'STEP':
            last_step_index = i

    # check for NOTE duplication
    for event in previous_events[last_step_index + 1:]:
        if event == sampled_event:
            print('Note duplication: resample')
            return False

    return True


def is_valid_event(sampled_event):
    """
    Checks if a given event is a valid MIDI event.
    """
    valid_events = {'NOTE', 'STEP', 'DELTA', 'EVEN', 'TRIPLE', 'MEASURE', '[PAD]'}
    event_prefix = sampled_event.split('=')[0]

    return event_prefix in valid_events


def syntax_check(sampled_token, previous_tokens, tokenizer):
    """
    Takes a sampled token and a list of tokens that come before it, and checks if appending the former to the latter results in a syntactically valid token sequence.
    Returns a boolean value indicating whether a token is valid (True) or should be resampled (False).
    """
    sampled_event = tokenizer.reverse_vocab[sampled_token]
    print(f'sampled token: {sampled_event}')
    previous_events = tokenizer.detokenize(previous_tokens)

    if not is_valid_event(sampled_event):
        return False
    else:
        is_valid = all([
          fits_previous_token(sampled_event, previous_events),
          no_note_duplication(sampled_event, previous_events), # we could add more syntax check functions here
        ])
        return is_valid


# not in use, the model seems to predict measures well and it makes more sense to check for errors at the sequence level, not the token level
def measure_check(sampled_token, previous_tokens, tokenizer):
    """
    Checks if the sampled event is a STEP DELTA event that moves time forward to or beyond the next measure start.
    If it isn't, returns False. If it is, returns [(corrected) DELTA, MEASURE]
    """
    sampled_event = tokenizer.reverse_vocab[sampled_token]
    previous_events = tokenizer.detokenize(previous_tokens)

    if sampled_event.startswith('DELTA') and previous_events[-1] == 'STEP':
        total_sequence = previous_events + [sampled_event]

        next_measure = 0
        current_time = 0
        for index, event in enumerate(total_sequence):
            if event == 'EVEN':
                next_measure += 4
            if event == 'TRIPLE':
                next_measure += 3
            if event == 'STEP':
                # assumes that STEPs are followed by DELTAs
                delta = total_sequence[index+1]
                delta = float(delta.split('=')[1])
                current_time += delta

        if current_time == next_measure:
            print('Inserted MEASURE')
            return [sampled_event, 'MEASURE']
        if current_time > next_measure:
            adjusted_delta = f'DELTA={delta - (current_time - next_measure)}'
            print(f'Inserted MEASURE and adjusted DELTA to {adjusted_delta}')
            return [adjusted_delta, 'MEASURE']
    return False


##### SEQUENCE GENERATION #####

def sample_token_with_syntax_check(predictions, previous_tokens, tokenizer, p=0.8):
    """
    Samples a token from given logits. Then applies a syntax check and, if necessary, resamples.
    If resampling doesn't result in an event that passes the syntax check after MAX_ATTEMPTS attempts, the threshold for top-p sampling is adjusted in a step-wise manner.
    """
    sampled_token = sample_top_p(predictions, p=p)

    MAX_ATTEMPTS = 40
    current_attempts = 0
    ith_probability_adjust = 0
    while not syntax_check(sampled_token, previous_tokens, tokenizer):
        sampled_token = sample_top_p(predictions, p=p)
        current_attempts += 1

        if current_attempts > MAX_ATTEMPTS:
            ith_probability_adjust += 1
            p += 1 / (ith_probability_adjust * 20)  # Increase p by a factor inversely proportional to the total number of probability adjustments
            current_attempts = 0

    return sampled_token


def generate_top_p(model, tokenizer, seed, max_len=100, p=0.8):

    # input is midi filepath
    if isinstance(seed, str):
        start_tokens = get_events(seed, round_values=True)
        start_tokens = tokenizer.tokenize(start_tokens)

    if isinstance(seed, list):
        # list of integer tokens
        if isinstance(seed[0], int):
            start_tokens = seed
        # list of str events
        elif isinstance(seed[0], str):
            start_tokens = tokenizer.tokenize(seed)
        else:
            raise ValueError("Input must be a MIDI filepath, a list of string tokens, or a list of integer tokens.")

    generated_tokens = start_tokens.copy()
    num_tokens = len(generated_tokens)

    while num_tokens < max_len:
        pad_len = max_len - num_tokens
        sample_index = num_tokens - 1

        if pad_len > 0:
            input_tokens = generated_tokens + [0] * pad_len
            input_tokens = np.array([input_tokens])
        else:
            raise ValueError('Input length should be less than max_len.')

        # model output has dimensions (1 (batch), SEQUENCE_LENGTH, vocab_size)
        predictions = model.predict(input_tokens, verbose=0)[:, sample_index, :] # take logits at sample_index
        predictions = predictions[0]

        sampled_token = sample_token_with_syntax_check(predictions, generated_tokens, tokenizer, p=p)

        # stop generating if padding token is predicted
        if tokenizer.reverse_vocab[sampled_token] == '[PAD]':
            break

        generated_tokens.append(sampled_token)
        num_tokens = len(generated_tokens)


    generated_sequence = tokenizer.detokenize(generated_tokens)
    return generated_sequence


def generate_midi(model, tokenizer, seed, max_len=100, p=0.8, tempo=120, save_midi=False, output_filepath=None):
    """
    Generate MIDI from a language model using top-p sampling.
    """
    generated_sequence = generate_top_p(model=model,
                                        tokenizer=tokenizer,
                                        seed=seed,
                                        max_len=max_len,
                                        p=p)

    midi_object = decode_tokens_to_midi(generated_sequence, tempo=tempo)

    if save_midi:
        midi_object.write(output_filepath)

    # removed: plot piano roll
    # removed: play midi with pygame

    return generated_sequence, midi_object