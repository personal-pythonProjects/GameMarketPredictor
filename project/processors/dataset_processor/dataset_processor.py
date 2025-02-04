import pandas as pd
from abc import ABC, abstractmethod

class DatasetProcessor(ABC):
    def __init__(self, file_path):
        self._file_path = file_path
        self._data = None

    def _load_dataset(self):
        try:
            self._data = pd.read_csv(self._file_path)

            if self._data.empty:
                raise ValueError(f'Dataset is empty! Make sure {self._file_path} contains data.')

        except FileNotFoundError:
            print(f'Error: File {self._file_path} not found. Ensure the file is in the correct directory.')
        except pd.errors.ParserError:
            print(f'Error: Failed to read the file. Check your CSV format.')
        except ValueError as e:
            print(f'Error: {e}')

    def _drop_null_values(self):
        try:
            if self._data is None:
                raise ValueError('Invalid dataset! Ensure the data is successfully loaded')

            self._data = self._data.dropna()

            if self._data.empty:
                raise ValueError('All rows contain missing values! The dataset becomes empty after dropping nulls.')

        except ValueError as e:
            print(f'Error: {e}')

    def process_data(self):
        self._load_dataset()
        self._drop_null_values()

    def get_data(self):
        return self._data

    @abstractmethod
    def analyze(self):
        pass







