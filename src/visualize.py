import seaborn as sns
import matplotlib.pyplot as plt

def plot_pairplot(df):
    sns.pairplot(df, hue='target')
    plt.show()
