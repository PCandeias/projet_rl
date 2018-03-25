from pathlib import Path

models_directory = "models/"
EPS = 1e-8

def file_exists(filename):
    file = Path(filename)
    return file.is_file()
