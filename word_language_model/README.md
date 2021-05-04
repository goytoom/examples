# Changes

- Added data folder `positive` from task 1
- Added models folder `models` with model `model_pos` from task 1 (trained on positive reviews)
- Added `scripts` folder with `train_new.sh` bash that can be used and modified instead of the command line. The sample will be saved under `samples`
- Modified `generate.py` to accept an input sequence, feed it to the hidden layer and generate a text starting after the sequence. Comments in code file!

# Work flow

Generate a sample text by typing `python3 generate.py --input "your string"`.
The file will be named `generated.txt`.
The script will work with my model from task 1 and the respective data set.
