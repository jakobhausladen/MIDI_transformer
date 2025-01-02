# Lute Music MIDI Transformer



## Overview

The goal of this project was to train a generative model on MIDI data to produce lute music in historical style. 

## Data

The data stems from Sarge Gerbode's fantastic [website](https://wp.lutemusic.org), who has collected and transcribed thousends of pieces of lute music to make them accessible to music performers. For the present model, I have focused on English lute music from the late 16th and early 17th century, to obtain a stylistically coherent selection.

The MIDIs are translated into 6-measure long token sequences consisting of the tokens: 
- `MEASURE`, `EVEN`, `TRIPLE`, `NOTE=<PITCH>`, `STEP`, `DELTA=<TIME>`.

## Data Augmentation

I have effectively doubled the data by transposing the music into neighboring keys.

## Model

The models were created and trained with `keras_nlp`. After initial attempts with LSTMs, I have settled on the decoder-only tranformer architecture (essentially, a miniature GPT). The final model has the following hyperparameters:

- Embedding dimension: 84 (reflecting the relatively small token vocabulary compared to typical NLP tasks).
- Feed forward dimension: 320.
- Layers: 10.
- Attention heads: 10.
- Dropout: 0.1.


This gives us roughly 850,000 trainable parameters. The model was trained using categorical cross-entropy as loss function, a batch size of 64, and the adam optimizer. It was trained on NVIDIA T4 GPUs using `Google Colab`.

## Results

Training was stopped after 120 epochs, before the model had fully converged. Dropout was successful in preventing overfitting, which had beed a problem with earier models.

![alt text](training_history-1.png)

Below, you find a particularily nice example of a 6-measure sequence, generated using the first 1.5 measures of John Dowland's *Lachrimae* as prompt. The corrsponding MIDI `lachimae_nice.mid` can be found in the top-level directory of this repository, if you want to take a listen.

![alt text](Capture.PNG)

### Longer Pieces

I have experimented with generating longer pieces by chaining a moving window of 6 measures. But while the 6-measure snippets were both coherent and musically interesting, the longer pieces quickly lost focus and ended in boring and repetative musical structures.

### Future Work

There are several parameters that could be further experimented with to improve performance.

- Amount of data: I have used only a fraction of the data to have a stylistically consistent subset. An alternative would be to use the entire data and then fine-tune on the subset.
- Encoding: Different encoding strategies and techniques such as byte-pair encoding have been shown to affect the modeling results.

## Repository Contents

- **Data**:
  - MIDI files of Renaissance lute compositions.
  - Metadata for MIDIs.
  - Training data:
    - Token sequences generated from MIDIs, with different measure counts.
    - Versions with and without augmentation (transposition).

- **`midi_utils`**:
  Contains modules for processing, analyzing, and generating MIDI data:
  - **`display`**:
    - Plot piano rolls.
  - **`encoding_decoding`**:
    - Extract note events from MIDIs and encode them as token sequences:
    - Parse and encode a directory of MIDIs and compute musical feature statistics.
    - Decode token sequences and generate MIDI files.
  - **`preprocessing`**:
    - Tokenizer class for numeric representation.
    - Split token sequences into n-measure chunks.
    - Transpose sequences for data augmentation.
  - **`midi_generation`**:
    - Perform top-p sampling for MIDI generation.
    - Run extensive syntax checks on generated token sequences.
    - Generate token sequences and MIDI files using a model and prompt.
  - **`train`**:
    - Prepare training data.
    - Initialize a transformer model with specified hyperparameters.
    - Train the model and generate example outputs.

- **Notebooks**:
  - **`data_preparation`**:
    - Select and clean a subset of the data using metatdata and musical feature statistics.
    - Create training datasets.
  - **`train`**:
    - Initialize or load a model.
    - Train the model.
  - **`examples`**:
    - Generate longer musical examples with measure checking.
  - **`plotting`**:
    - Visualize generated examples and training history.

- **`models`**:
  - Stores experiments, including model files, training histories, and test runs.

