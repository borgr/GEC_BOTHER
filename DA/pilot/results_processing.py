import pandas as pd
import numpy as np

RESULTS_FILE_ADDR = "Batch_3548123_batch_results.csv"


# gets an HIT id and return an array of size 15 with the indexes of the perfect sentences in the hit
def get_p_indexes(HIT_id):
    pass


# return an array which contains this c_sentence ranks (array size depends on the number of annotators for the sentence)
def c_sentence_scores(id):
    pass  # TODO:  this.


# return an array of size 100 which contains this HIT scores
def HIT_scores(HIT_id):
    res = []
    for i in range(1, HIT_SIZE + 1):
        # add this HIT score for sentence i to res.
        pass


# return an array of size 15 which contains this HIT perfect sentences scores
def HIT_p_scores(HIT_id):
    pass  # TODO:  this.


def c_sentence_average(id):
    return average(c_sentence_scores(id))


def HIT_average(HIT_id):
    return average(HIT_scores(id))


def HIT_p_average(HIT_id):
    return average(HIT_p_scores(id))


# Python program to get average of a list
def average(lst):
    return sum(lst) / len(lst)


def main():
    data = pd.read_csv(RESULTS_FILE_ADDR, index_col=0)
    print(data.tail())



if __name__ == "__main__":
    main()
