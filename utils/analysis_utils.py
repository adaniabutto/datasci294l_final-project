import nltk
import numpy as np
from math import log
import pandas as pd


def replace_letters(text, letter1, letter2):
    """Replaces all occurrences of letter1 with letter2 and vice versa in the given text."""
    
    temp_char = "#" # Using a temporary character to avoid conflicts during replacement
    text = text.replace(letter1, temp_char)
    text = text.replace(letter2, letter1)
    text = text.replace(temp_char, letter2)
    return text
    
def get_freq_profile(tokens, n_value=2):
    """
    Compute the frequency of n-grams in a given list of tokens.
    Args:
        tokens (list): The input list of tokens.
        n_value (int, optional): The value of n for n-grams. Defaults to 2.
    Returns:
        nltk.FreqDist: A frequency distribution of n-grams.
    """

    ngrams = nltk.ngrams(tokens, n_value)
    ngram_fdist = nltk.FreqDist(ngrams)
    return ngram_fdist


def convert_fd_to_pd(FD):
    """
    convert a frequency dictionary to a probability dictionary
    Args:
        FD (nltk.FreqDist): The frequency dictionary.
    Returns:
        dict: A dictionary with relative frequencies.
    Example:
        >>> FD = nltk.FreqDist({'a': 3, 'b': 2, 'c': 1})
        >>> convert_fd_to_pd(FD)
        [0.5, 0.33, 0.17]
    """
    total = float(sum(FD.values()))
    rel_freq = [x / total for x in FD.values()]
    return dict(zip(FD.keys, rel_freq))


def entropy(probabilities):
    """
    compute the entropy of a probability distribution
    """
    res = 0.0
    for x in probabilities:
        res += x * log(x, 2)
    return -res


def compute_entropy(count_dictionary, type="freq"):
    """
    compute the entropy of distribution
    """
    if type == "freq":
        PD = convert_fd_to_pd(count_dictionary)
        return entropy(PD.values())
    elif type == "prob":
        return entropy(count_dictionary.values())
    else:
        raise ValueError("type should be freq or prob")


def sort_freq_dict(FD, descending=True):
    """
    Sort a frequency dictionary by values in descending order.
    Args:
        freq_dict (dict): The frequency dictionary to sort.
    Returns:
        list: A sorted list of tuples (key, value) in descending order.
    Example:
        >>> freq_dict = {'a': 3, 'b': 2, 'c': 1}
        >>> sort_freq_dict(freq_dict)
        [('a', 3), ('b', 2), ('c', 1)]
    """
    from operator import itemgetter

    ngrams = [(" ".join(ngram), FD[ngram]) for ngram in FD]
    return sorted(ngrams, key=itemgetter(1), reverse=descending)


def countBoxOnGoal(listOfBoxes):
    """
    count the number of boxes on goal
    Args:
        listOfBoxes (list): A list of box dictionaries.
    Returns:
        int: The count of boxes that are on goal.
    Example:
        >>> myboxes = [{'onGoal': True}, {'onGoal': False}, {'onGoal': True}]
        >>> countBoxOnGoal(myboxes)
        2
    """
    return sum(box["onGoal"] for box in listOfBoxes)


def maxBoxOnGoal(listOfEvents):
    allboxes = [countBoxOnGoal(event.get("boxes")) for event in listOfEvents]
    return max(allboxes, default=pd.NA)


def fill_with_function(df, target_col, ref_col, func, initial_value=None, **kwargs):
    """
    Fills values in a DataFrame column using a function
    that combines the previous row's value and another column's value.

    Args:
        df (pd.DataFrame): The DataFrame.
        target_col (str): The name of the column with NaN values to fill.
        ref_col (str): The name of the column to use in the function.
        func (callable): A function that takes two arguments
                         (previous value, reference column value)
                         and returns a new value.
        initial_value: The value to use for the first row if empty
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        pd.DataFrame: The DataFrame with NaN values filled.
    """
    previous_value = initial_value
    for index, row in df.iterrows():
        df.at[index, target_col] = func(previous_value, row[ref_col], **kwargs)
        previous_value = df.at[index, target_col]
    return df


# if __name__ == "__main__":
#     arrows = "lruddurldurld"
#     ngram_window = 2
#     freq_dist = get_freq_profile(arrows, ngram_window)

#     # Test printing the frequency distribution
#     # print(freq_dist.most_common(10))
#     for ngram in list(freq_dist.items())[:12]:
#         print(" ".join(ngram[0]), ngram[1])
#     print("...")

#     # Test the sorting function
#     sorted_freq_dist = sort_freq_dict(freq_dist)
#     for t in sorted_freq_dist[:12]:
#         print(t[0], t[1])

def count_consecutive_changes(numbers):
    if len(numbers) < 2:
        return {'increase': 0, 'decrease': 0, 'same': 0}
    
    changes = {'increase': 0, 'decrease': 0, 'same': 0}
    
    for i in range(1, len(numbers)):
        diff = numbers[i] - numbers[i-1]
        if diff > 0:
            changes['increase'] += 1
        elif diff < 0:
            changes['decrease'] += 1
        else:
            changes['same'] += 1
            
    return changes

 # Test cases
# print(count_consecutive_changes([1,2,3,2,2,1,0]))  # {'increase': 2, 'decrease': 3, 'same': 1}
# print(count_consecutive_changes([1,1,1,1]))        # {'increase': 0, 'decrease': 0, 'same': 3}
# print(count_consecutive_changes([1]))              # {'increase': 0, 'decrease': 0, 'same': 0}
# print(count_consecutive_changes([]))               # {'increase': 0, 'decrease': 0, 'same': 0}
# print(count_consecutive_changes([5,4,3,2,1]))      # {'increase': 0, 'decrease': 4, 'same': 0}
