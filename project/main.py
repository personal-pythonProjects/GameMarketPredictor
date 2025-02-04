from project.processors.correlation_analyzer.heatmap_correlation_analyzer import HeatmapCorrelationAnalyzer
from project.processors.model_trainer.linear_regression_trainer import LinearRegressionTrainer
from project.processors.sales_analyzer.genre_sales_analyzer import GenreSalesAnalyzer
from constants.constants import FILE_PATH
import re


def get_valid_mode_name():
    while True:
        try:
            user_input = input('Enter the model name: ').strip().lower()

            if not user_input:
                raise ValueError('Model cannot be empty.')

            if not re.match(r'^[a-z\s]+$', user_input):
                raise ValueError('Model name should only contain letters and spaces.')

            model_name = '_'.join(user_input.split())

            return model_name

        except ValueError as e:
            print(f'Error: {e}')


if __name__ == "__main__":

    # Genre Sales Analyzer
    analyzer = GenreSalesAnalyzer(FILE_PATH)
    analyzer.process_data()
    analyzer.analyze()

    # Heatmap Correlation Analyzer
    correlation_matrix = HeatmapCorrelationAnalyzer(FILE_PATH)
    correlation_matrix.process_data()
    correlation_matrix.analyze()

    # Linear Regression Trainer
    linear_regression_trainer = LinearRegressionTrainer(FILE_PATH, get_valid_mode_name())
    linear_regression_trainer.process_data()
    linear_regression_trainer.analyze()