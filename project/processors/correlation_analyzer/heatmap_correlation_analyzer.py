import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from project.processors.dataset_processor.dataset_processor import DatasetProcessor

class HeatmapCorrelationAnalyzer(DatasetProcessor):

    def __init__(self, file_path):
        super().__init__(file_path)

    def analyze(self):
        try:
            data = self.get_data()

            if data is None:
                raise ValueError('Invalid dataset! Cannot analyze correlation.')

            numerical_data = data.select_dtypes(include=[np.number])

            if numerical_data.empty:
                raise ValueError('No numerical columns found in the dataset.')

            correlation_matrix = numerical_data.corr()

            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, cmap='YlOrBr', annot=True)
            plt.title('Heatmap Correlation between Variable')
            plt.show()

            print("Correlation Matrix:")
            print(correlation_matrix)

        except ValueError as e:
            print(f'Error: {e}')
        except Exception as e:
            print(f'Unexpected error occurred: {e}')