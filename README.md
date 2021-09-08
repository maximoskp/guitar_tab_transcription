# guitar_tab_transcription
Transcribing guitar tabs from various musical inputs

## Environment setup
create conda environment:
conda env create -f guitar_tab.yml

save conda environment, if you make changes (caution: remove os and cuda-related dependencies!):
conda env export -n guitar_tab -f guitar_tab.yml --no-builds

## Before you run
Create folder structure (todo, with python script)
Before running again, make sure folders are empty (reset with python script)

What works best:
run_tab_flat_ANN.py

How to run:
run_make_events.py
run_make_pianoroll_tab.py
run_tab_flat_ANN.py
