import plotly.graph_objects as go
import pretty_midi
from midi_utils.encoding_decoding import get_note_events


def plot_piano_roll(midi_file_path, seed_length=None, width=800, height=400):

    tempi, pitch_counts, note_events = get_note_events(midi_file_path)
    measures = [note_event for note_event in note_events if note_event[0] == 0]
    note_events = [note_event for note_event in note_events if note_event[0] != 0]
    
    x_min = min([note_event[1] for note_event in note_events])
    x_max = max([note_event[1] for note_event in note_events]) + note_events[-1][2]
    y_min = min([note_event[0] for note_event in note_events]) - 2.5
    y_max = max([note_event[0] for note_event in note_events]) + 2.5

    
    fig = go.Figure()

    # add barlines
    for i in [measure[1] for measure in measures][1:]:
        fig.add_shape(
            type='line',
            x0=i,
            y0=y_min,
            x1=i,
            y1=y_max,
            line=dict(
                color='#1258DC',
                width=1,
                dash='dot'
            )
        )

    # add notes
    for index, note_event in enumerate(note_events):

        # color the first n=seed_length notes purple
        if not seed_length is None and index+1 <= seed_length:
            color = '#3A004C'
            fillcolor = '#C91BFE'
        else:
            color = '#0A337F'
            fillcolor = '#6395F2'
    
        pitch = note_event[0]
        note_name = pretty_midi.note_number_to_name(pitch)
        note_start = note_event[1]
        duration = note_event[2]
        note_end = note_start + duration

        x0 = note_start
        y0 = pitch-0.5
        x1 = note_end
        y1 = pitch+0.5
        
        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[y0, y0, y1, y1, y0],
            mode='lines',
            fill='toself',
            line=dict(color=color, width=1),
            fillcolor=fillcolor,
            text=note_name,
            hoverinfo='text'
        ))


    # set axes ranges and grid
    fig.update_xaxes(
        range=[x_min-0.02, x_max+0.02],
        minallowed=x_min-0.02,
        maxallowed=x_max+0.02,
        showgrid=True,
        gridwidth=1,
        dtick=1,
        showticklabels=False
    )
    fig.update_yaxes(
        range=[y_min, y_max],
        minallowed=y_min,
        maxallowed=y_max,
        showgrid=True,
        gridwidth=1,
        tick0=0.5,
        dtick=1,
        showticklabels=False
    )

    
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=False,
        template='plotly_white'
    )

    config = {'scrollZoom': False,
              'displaylogo': False,
              'modeBarButtonsToRemove': ['zoomIn', 'zoomOut', 'autoScale', 'toImage']}
    
    fig.write_html('piano_roll.html', config=config)
    fig.show(config=config)