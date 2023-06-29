import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

class MarketBasketAnalysis:
    """Class for performing market basket analysis."""
    def __init__(self, file, country, item_name):
        """Initialize the class with the input file, country, and item_name."""
        self.file = file
        self.country = country
        self.item_name = item_name
        self.df = pd.read_csv(self.file)

    def preprocess_data(self):
        """
        Preprocesses the data: drops NaN in 'Description' and 'Invoice' column,
        and filters rows where 'Quantity' is greater than 0.
        """
        self.df = self.df.dropna(subset=['Description', 'Invoice'])
        self.df = self.df[self.df['Quantity'] > 0]

    def create_basket(self):
        """
        Creates a basket by grouping by 'Invoice' and 'Description' and setting 'Invoice' as index.
        """
        self.basket = (self.df[self.df['Country'] == self.country]
                       .groupby(['Invoice', 'Description'])['Quantity']
                       .sum().unstack().reset_index().fillna(0)
                       .set_index('Invoice'))

    @staticmethod
    def encode_units(x):
        """
        Encodes the units to True if the value is >= 1 and False otherwise.
        """
        return x >= 1

    def generate_frequent_itemsets(self):
        """
        Generates frequent itemsets using apriori algorithm.
        """
        basket_sets = self.basket.applymap(self.encode_units)
        self.frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

    def generate_rules(self):
        """
        Generates association rules from the frequent itemsets.
        """
        self.rules = association_rules(self.frequent_itemsets, metric="lift", min_threshold=1)

    def filter_rules(self):
        """
        Filters rules based on a given item in the antecedents.
        """
        self.filtered_rules = self.rules[self.rules['antecedents'].apply(lambda x: self.item_name in x)]

    def get_possible_items(self):
        """
        Gets the possible items that can be associated with a given item.
        """
        self.possible_items = self.filtered_rules['consequents'].apply(lambda x: list(x)[0])
        print(self.possible_items)

    def run_analysis(self):
        """Execute the full analysis."""
        self.preprocess_data()
        self.create_basket()
        self.generate_frequent_itemsets()
        self.generate_rules()
        self.filter_rules()
        self.get_possible_items()


if __name__ == "__main__":
    file = 'online_retail_II.csv'
    country = "Germany"
    item_name = "PLASTERS IN TIN WOODLAND ANIMALS"

    mba = MarketBasketAnalysis(file, country, item_name)
    mba.run_analysis()
