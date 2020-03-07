import matplotlib.pyplot as plt
import pandas as pd


def distribution(df, criteria):
    """Show table distribution by some criteria

    :param df: dataframe
    :param criteria: dataframe column name to group values by
    :return: dataframe
    """
    return df.groupby([criteria]).count()


def topic_distribution(df, **kwargs):
    """Show barplot with number of texts per topic

    :param df: dataframe
    :param kwargs: output_path - path to save the plot; name - title for the plot
    :return:
    """
    dt = pd.concat([df], axis=0)
    dt.category.value_counts().plot(figsize=(15, 10), kind='bar', color='green')
    plt.xlabel('Topic')
    plt.ylabel('Number of Items Per Topic')
    plt.subplots_adjust(top=0.9, bottom=0.28)
    if 'name' in kwargs.keys():
        plt.title(kwargs['name'])
    if 'output_path' in kwargs.keys():
        plt.savefig(kwargs['output_path'])
    else:
        plt.show()
