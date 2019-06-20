import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from my_parser import NUCLE_DB_ADDR, parse_sentence, MISTAKES_INX
from results_processing import get_all_z_scores, RESULTS_FILE_ADDR

MISTAKE_TYPES = ["Vt", "Vm", "V0", "Vform", "SVA", "ArtOrDet", "Nn", "Npos", "Pform", "Pref", "Prep", "Wci",
                 "Wa", "Wform", "Wtone", "Srun", "Smod", "Spar", "Sfrag", "Ssub", "WOinc", "WOadv", "Trans",
                 "Mec", "Rloc-", "Cit", "Others", "Um"]
EVALUATION_FEATURES = ["Nucle_ID"] + MISTAKE_TYPES + ["TotalMistakes", "z-score"]
TO_MTURK_FILE_PATH = r"C:\Users\ofir\Documents\University\year2\GEC Project\GEC_ME_PROJECT-master\GEC_ME_PROJECT\NUCLE\toMturk\mTurk_csv.csv"


def get_senteces(nucle_addr,to_mturk_csv_addr,results_from_mturk_addr):
    lines = open(nucle_addr).read().splitlines()
    hits = pd.read_csv(to_mturk_csv_addr)
    sentences = pd.DataFrame(columns= EVALUATION_FEATURES)
    z_scores = get_all_z_scores(results_from_mturk_addr)
    for index, row in hits.iterrows():
        for i in range(1, 101):
            sentences.loc[i + 100*index] = [0]*len(EVALUATION_FEATURES)
            id = row["Nucle_ID" + str(i)] -1
            mistakes = parse_sentence(lines, id)[MISTAKES_INX]
            sentences.loc[i + 100*index]["Nucle_ID"] = id
            for mistake in mistakes:
                sentences.loc[i + 100 * index][mistake[2]] += 1
                sentences.loc[i + 100 * index]["TotalMistakes"] += 1
            sentences.loc[i + 100 * index]["z-score"] = z_scores.loc[index]["Answer.mistakeRate" + str(i)]
    return sentences


def mistakes_stats(sentences):
    df = pd.DataFrame(columns= ["number_of_sentences","perc_of_sentences", "total_appearance", "perc_of_appearance"])
    sentences = get_X(sentences)
    df["number_of_sentences"] = sentences.astype(bool).sum(axis=0)
    df["total_appearance"] = sentences.sum(axis=0)
    df["perc_of_sentences"] =  df["number_of_sentences"]/sentences.shape[0]
    df["perc_of_appearance"] =  df["total_appearance"]/df["total_appearance"]["TotalMistakes"]
    return df


def get_X(sentences):
    return sentences.drop('z-score',1).drop('Nucle_ID',1)

if __name__ == '__main__':
    sentences = get_senteces(NUCLE_DB_ADDR, TO_MTURK_FILE_PATH,RESULTS_FILE_ADDR)
    # mistakes_stats(sentences)
    y = np.array(sentences["z-score"])
    X = np.array(get_X(sentences))
    #y = 1 * x_0 + 2 * x_1 + 3
    print(get_X(sentences).shape)
    # print(y)
    reg = LinearRegression()
    reg.fit(X, y)
    print(len(reg.coef_), reg.coef_)
    print(reg.intercept_ )

