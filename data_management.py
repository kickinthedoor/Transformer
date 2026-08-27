import os
import shutil


def reset_dataset_cache(folders=('./data', './cache', './models'), extensions=('.model', '.vocab')):
    """
    Clears cached raw data, tokenized cache, and tokenizer model/vocab files
    so the next prepare_data() call rebuilds everything from scratch.
    """
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    for f in os.listdir('.'):
        if f.endswith(extensions):
            os.remove(f)

    print("Old cache and tokenizers cleared! Ready to run prepare_data().")
