import matplotlib.pyplot as plt
import matplotlib as mpl
from project.processors.dataset_processor.dataset_processor import DatasetProcessor

class GenreSalesAnalyzer(DatasetProcessor):
    def __init__(self, file_path):
        super().__init__(file_path)

    def analyze(self):
        try:
            data = self.get_data()

            if data is None:
                raise ValueError('Invalid dataset! Cannot generate the pie chart.')

            if 'Genre' not in data.columns or 'Global_Sales' not in data.columns:
                raise KeyError('Column \'Genre\' or \'Global_Sales\' is missing from dataset.')

            game_sales = data.groupby('Genre')['Global_Sales'].sum().head(10)

            if game_sales.empty:
                raise ValueError('Not enough data to generate the pie chart.')

            custom_colors = mpl.colors.Normalize(vmin=min(game_sales), vmax=max(game_sales))
            colours = [mpl.cm.PuBu(custom_colors(i)) for i in game_sales]

            plt.figure(figsize=(7, 7))
            plt.pie(game_sales, labels=game_sales.index, colors=colours, autopct='%1.1f%%')
            central_circle = plt.Circle((0, 0), 0.5, color='white')

            fig = plt.gcf()
            fig.gca().add_artist(central_circle)

            plt.rc('font', size=12)
            plt.title('Top 10 Game Categories by Sales', fontsize=20)
            plt.show()

        except KeyError as e:
            print(f'Error: {e}')
        except ValueError as e:
            print(f'Error: {e}')
        except Exception as e:
            print(f'Unexpected error occurred: {e}')