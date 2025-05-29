MIDI Music Generator with Python GUI

This project provides a Python application with a graphical user interface (GUI) based on Tkinter to generate MIDI files using a pre-trained Seq2Seq Transformer model. The application allows users to select various musical metadata (such as key, instrument, tempo) or let them be chosen randomly, and to configure generation parameters like temperature and desired track length.
Description

The application loads a deep learning model trained for music generation and its corresponding MIDI and metadata vocabularies. Through an intuitive interface, the user can specify the characteristics of the MIDI track they wish to create. The generation process is managed in a separate thread to keep the GUI responsive. The resulting MIDI files are saved in a specified directory.
Main Features

    Graphical User Interface (GUI): Simple and intuitive, built with Tkinter.

    Custom Metadata Selection:

        Key

        Instrument

        Time Signature

        Tempo (BPM)

        Average Velocity

        Velocity Range

        Number of Instruments

    Random Selection: Option to let the application randomly choose metadata for each category.

    Configurable Generation Parameters:

        Temperature: Controls the randomness of the model's output.

        Target MIDI Length: Specifies the desired length of the generated track in number of tokens.

    Multiple Generation: Generates 3 MIDI files per session with one click.

    Integrated Logging: Displays status and error messages directly in the interface.

    Asynchronous Generation: The generation process runs in a separate thread to avoid blocking the GUI.

    Transformer-Based: Uses a Seq2Seq Transformer model for MIDI sequence generation.

Prerequisites

Before running the application, ensure you have Python (version 3.7 or higher recommended) and the following libraries installed:

    torch (PyTorch)

    miditok

    symusic

    tkinter (usually included in standard Python installations)

You can install missing dependencies using pip:

pip install torch miditok symusic

(Ensure you install the PyTorch version appropriate for your system and any CUDA support: https://pytorch.org/)
Configuration and Installation

    Clone the Repository (or download the files):

    git clone https://github.com/mbiggeri/PianoGenerator.git

    Download the Pre-trained Model and Vocabularies:

        Ensure you have the model checkpoint file (.pt), MIDI vocabulary (midi_vocab.json), and metadata vocabulary (metadata_vocab.json).

        Update the paths within the Python script (music_generator_gui.py or your script's name) to the correct locations of these files on your system:

        PATH_MODELLO_CHECKPOINT = Path(r"PATH_TO_YOUR_MODEL.pt")
        PATH_VOCAB_MIDI = Path(r"PATH_TO_YOUR_MIDI_VOCAB.json")
        PATH_VOCAB_METADATA = Path(r"PATH_TO_YOUR_METADATA_VOCAB.json")
        GENERATION_OUTPUT_DIR = Path("./generated_midi_output") # Or your preferred folder

    Install Dependencies:
    If you haven't already, install the libraries listed in the "Prerequisites" section. You might want to create a virtual environment:

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt # If you provide a requirements.txt file

    Or install manually as indicated above.

Usage

    Run the Python script from the command line:

    python music_generator_gui.py

    (Replace music_generator_gui.py with the actual name of your script file).

    The application window will open.

    Select Metadata: For each category (Key, Instrument, etc.), choose a specific token from the dropdown menu or leave the "Random" option.

    Set Generation Parameters:

        Enter the desired value for Temperature (e.g., 0.75).

        Enter the Target MIDI Length in tokens (e.g., 2048).

    Click the "Generate 3 MIDI" button.

    The application will begin the generation process. You can monitor the status and any messages in the log area at the bottom.

    The generated MIDI files will be saved in the directory specified by GENERATION_OUTPUT_DIR (default ./generated_midi_inference_gui or as configured). Each filename will include a timestamp and some of the metadata used for generation.

Project Structure (Example)

PianoGenerator/
│
├── music_generator_gui.py     # The main GUI application script
├── model_checkpoint.pt      # (Example) Your pre-trained model (to be placed according to paths)
├── midi_vocab.json            # (Example) Your MIDI vocabulary (to be placed)
├── metadata_vocab.json        # (Example) Your metadata vocabulary (to be placed)
├── generated_midi_output/     # Output folder (created automatically)
│   └── ... (generated MIDI files)
└── README.md                  # This file

Note: The model and vocabulary files are not included in this repository and must be obtained/placed separately. Update the paths in the script as indicated.
Known Issues / Limitations

    Initial model loading might take a few seconds, during which the GUI might seem unresponsive if the model is very large and not handled with more advanced startup loading.

    Input validation for Temperature and Target Length is basic.

    The application has been primarily tested on [specify your environment if relevant, e.g., Windows 10, Python 3.9].

Contributing

Contributions are welcome! If you want to contribute, please open an issue to discuss the proposed changes or submit a Pull Request.
License
