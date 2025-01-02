import numpy as np
import pretty_midi

import json
import os
from tqdm import tqdm


##### ENCODING #####

def round_to_valid(num):
    closest_multiple = round(num / 0.0625) * 0.0625
    return round(closest_multiple, 4)


def get_note_events(midi_path, round_values=False):
    """
    Extracts note and downbeat events from a MIDI file.
    Note events are in the format [note.pitch, note.start, note_duration], downbeat events in the format [0, start, inf, measure_type].
    """
    def seconds_to_quarter_notes(time_seconds, tempo_change_times, tempi):
        time_quarter = 0
        tempo = tempi[0]
        last_tempo_change_time = 0
        for i in range(len(tempo_change_times)):
            if time_seconds >= tempo_change_times[i]:
                time_quarter += (tempo_change_times[i] - last_tempo_change_time) * tempo / 60
                tempo = tempi[i]
                last_tempo_change_time = tempo_change_times[i]
        time_quarter += (time_seconds - last_tempo_change_time) * tempo / 60
        return time_quarter

    midi_object = pretty_midi.PrettyMIDI(midi_path)
    tempo_change_times, tempi = midi_object.get_tempo_changes()
    tempi = np.round(tempi)

    note_events = []
    pitch_counts = {}

    # append note events in the format [pitch, note.start, note_duration]
    for instrument in midi_object.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note_start = seconds_to_quarter_notes(note.start, tempo_change_times, tempi)
                note_start = round_to_valid(note_start) if round_values else note_start
                note_end = seconds_to_quarter_notes(note.end, tempo_change_times, tempi)
                note_end = round_to_valid(note_end) if round_values else note_end
                note_duration = note_end - note_start
                note_duration = round_to_valid(note_duration) if round_values else note_duration
                note_pitch = note.pitch

                note_events.append([note_pitch, note_start, note_duration])
                pitch_counts[note_pitch] = pitch_counts.get(note_pitch, 0) + 1

    # append downbeat events in the format [0, start, inf, measure_type] where 0 and inf guarantee that they are sorted to the beginning of each measure
    downbeats = midi_object.get_downbeats()
    time_signatures = midi_object.time_signature_changes

    for downbeat in downbeats:
        # determine whether the measure is in even or triple meter
        measure_type = None
        for time_signature in time_signatures:
            if downbeat >= np.round(time_signature.time, 4):
                measure_type = 'even' if time_signature.numerator % 2 == 0 else 'triple'
            else:
                break
        downbeat = seconds_to_quarter_notes(downbeat, tempo_change_times, tempi)
        downbeat = np.round(downbeat, 4)
        note_events.append([0, downbeat, float('inf'), measure_type])

    # sort by (1) note.start (ascending), (2) duration (descending), and (3) pitch (ascending)
    note_events_sorted = sorted(note_events, key=lambda x: (x[1], -x[2], x[0]))
    return tempi, pitch_counts, note_events_sorted


def get_events(midi_path, round_values=False):
    """
    Extracts musical events from a MIDI file. Calls get_note_events and converts the output into token sequences.
    Tokens are of the form: 'MEASURE', 'EVEN', 'TRIPLE', 'NOTE=<PITCH>', 'STEP', 'DELTA=<TIME>'.
    """
    def represent_as_sum(value):
        """
        Could be used for splitting the DELTAs and reduce the number of unique events.
        Currently not in use. Not sure if this would result in a better or worse representation of the data.
        """
        factors = [2, 1, 0.75, 0.5, 0.25, 0.125, 0.0625]
        result = []

        for factor in factors:
            while value >= factor:
                result.append(factor)
                value -= factor

        return result


    tempi, pitch_counts, note_events = get_note_events(midi_path, round_values=round_values)

    events = []
    duration_counts = {}
    offset = 0

    for note_event in note_events:
        note_offset = round_to_valid(note_event[1]) if round_values else note_event[1]
        if note_offset != offset:
            time_step = note_offset - offset
            events.extend(['STEP', f'DELTA={time_step}'])
            offset = note_offset
        if note_event[2] == float('inf'):
            events.append('MEASURE')
            if note_event[3] == 'even':
                events.append('EVEN')
            else:
                events.append('TRIPLE')
            continue
        events.extend([f'NOTE={note_event[0]}', f'DELTA={note_event[2]}'])
        duration_counts[note_event[2]] = duration_counts.get(note_event[2], 0) + 1

    return tempi, pitch_counts, duration_counts, events


def parse_directory(input_directory, round_values=False, save_data=False, output_file_path=None):
    """
    Parses MIDI files in a directory, aggregates statistics, and optionally saves the data.
    For the format of the token sequences in the parsed MIDI data, see the documentation of get_events.
    """
    file_names = os.listdir(input_directory)
    parsed_midis = {}
    agg_tempo_counts = {}
    agg_pitch_counts = {}
    agg_duration_counts = {}

    for file_name in tqdm(file_names):
        file_path = os.path.join(input_directory, file_name)
        try:
            tempi, pitch_counts, duration_counts, events = get_events(file_path, round_values=round_values)
            parsed_midis[file_name] = {'pitch_counts': pitch_counts, 'duration_counts': duration_counts, 'tempi':tempi, 'events': events}
            for pitch, count in pitch_counts.items():
                agg_pitch_counts[pitch] = agg_pitch_counts.get(pitch, 0) + count
            for duration, count in duration_counts.items():
                agg_duration_counts[duration] = agg_duration_counts.get(duration, 0) + count

            for tempo in tempi:
                agg_tempo_counts[tempo] = agg_tempo_counts.get(tempo, 0) + 1
        except Exception as e:
            print(f"Error parsing {file_name}: {e}")

    data = {'agg_stats':{'pitch_counts':agg_pitch_counts, 'duration_counts':agg_duration_counts, 'tempo_counts':agg_tempo_counts}, 'parsed_midis':parsed_midis}

    if save_data:
        with open(output_file_path, 'wb') as f:
            json.dump(data, f)

    return data



##### DECODING #####

def decode_tokens_to_midi(event_list, tempo):
    """
    Decode a sequence of musical events represented as tokens into a PrettyMIDI object.
    The resulting PrettyMIDI object can be saved to a MIDI file using `midi_object.write('output.mid')`.
    """
    def to_seconds(duration, tempo=tempo):
        return duration / (tempo / 60)

    midi_object = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    midi_program = pretty_midi.instrument_name_to_program('Acoustic Guitar (nylon)')
    lute = pretty_midi.Instrument(program=midi_program, name='Lute')
    midi_object.instruments.append(lute)

    current_time = 0
    previous_element = None
    previous_time_signature = None
    note_list = []

    DEFAULT_VELOCITY = 64
    DEFAULT_DURATION = 0.5

    for event in event_list:
        if event == 'MEASURE':
            previous_element = 'MEASURE'
            continue

        if event == 'EVEN' and previous_element == 'MEASURE' and previous_time_signature != 'EVEN':
            midi_object.time_signature_changes.append(
                pretty_midi.TimeSignature(
                    numerator = 4,
                    denominator = 4,
                    time = current_time
                )
            )

        if event == 'TRIPLE' and previous_element == 'MEASURE' and previous_time_signature != 'TRIPLE':
            midi_object.time_signature_changes.append(
                pretty_midi.TimeSignature(
                    numerator = 3,
                    denominator = 4,
                    time = current_time
                )
            )

        elif event.startswith('NOTE'):
            _, pitch = event.split('NOTE=')
            pitch = int(pitch)
            note_on_time = current_time

            midi_object.instruments[0].notes.append(
                pretty_midi.Note(
                    velocity=DEFAULT_VELOCITY,
                    pitch=pitch,
                    start=note_on_time,
                    end=note_on_time + to_seconds(DEFAULT_DURATION, tempo)
                )
            )

            previous_element = 'NOTE'

        elif event.startswith('STEP'):
            previous_element = 'STEP'

        elif event.startswith('DELTA'):
            _, duration = event.split('DELTA=')
            duration = to_seconds(float(duration), tempo=tempo)

            if previous_element == 'NOTE':
                midi_object.instruments[0].notes[-1].end = current_time + duration

            if previous_element == 'STEP':
                current_time += duration

            previous_element = None

    return midi_object