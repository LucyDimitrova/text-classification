import matplotlib.pyplot as plt
import pandas as pd


def distribution(df, label):
    """Get number of examples per label

    :param df: dataframe
    :param label: dataframe column name to group values by
    :return: dataframe
    """
    return df.groupby([label]).count()


def label_distribution(df, label, **kwargs):
    """Show barplot with number of examples per label

    :param df: dataframe
    :param label: label column name
    :param kwargs: output_path - path to save the plot; name - title for the plot
    :return:
    """
    dt = pd.concat([df], axis=0)
    dt[label].value_counts().plot(figsize=(15, 10), kind='bar', color='yellow')
    plt.xlabel('Label')
    plt.ylabel('Number of Examples Per Label')
    plt.subplots_adjust(top=0.9, bottom=0.28)
    if 'name' in kwargs.keys():
        plt.title(kwargs['name'])
    if 'output_path' in kwargs.keys():
        plt.savefig(kwargs['output_path'])
    else:
        plt.show()
