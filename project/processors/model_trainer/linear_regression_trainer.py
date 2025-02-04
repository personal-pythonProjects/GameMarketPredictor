from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from project.processors.dataset_processor.dataset_processor import DatasetProcessor
import os

class LinearRegressionTrainer(DatasetProcessor):
    def __init__(self, file_path, model_name):
        super().__init__(file_path)
        self.__model_name = model_name
        self.__model_dir = 'models'
        self.__model = None
        self.__xtrain, self.__xtest, self.__ytrain, self.__ytest = None, None, None, None

    def _get_unique_filename(self):
        base_path = os.path.join(self.__model_dir, f'{self.__model_name}.pkl')
        if not os.path.exists(base_path):
            return base_path

        count = 1
        while True:
            unique_path = os.path.join(self.__model_dir, f'{self.__model_name}_new_{count}.pkl')
            if not os.path.exists(unique_path):
                return unique_path
            count += 1

    def analyze(self):
        try:
            data = self.get_data()

            if data is None:
                raise ValueError('Invalid dataset! Cannot train processors.')

            x = data[['Rank', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
            y = data['Global_Sales']

            self.__xtrain, self.__xtest, self.__ytrain, self.__ytest = train_test_split(x, y, test_size=0.2, random_state=42)

            self.__model = LinearRegression()
            self.__model.fit(self.__xtrain, self.__ytrain)

            if not os.path.exists(self.__model_dir):
                os.makedirs(self.__model_dir)

            model_path = self._get_unique_filename()

            with open(model_path, 'wb') as f:
                pickle.dump(self.__model, f)

            print(f'Linear Regression model saved to \'{model_path}\'.')

            # Evaluation model
            predictions = self.__model.predict(self.__xtest)
            mae = mean_absolute_error(self.__ytest, predictions)
            mse = mean_squared_error(self.__ytest, predictions)
            r2 = r2_score(self.__ytest, predictions)

            print('\nLinear Regression Performance')
            print('=' * 60)
            print(f'Mean Absolute Error: {mae}')
            print(f'Mean Squared Error: {mse}')
            print(f'R-squared: {r2}')

        except KeyError as e:
            print(f'Error: Column {e} not found in dataset.')
        except ValueError as e:
            print(f'Error: {e}')
        except Exception as e:
            print(f'Unexpected error occurred: {e}')