import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from my_parser import NUCLE_DB_ADDR, parse_sentence, MISTAKES_INX

RESULTS_FILE_ADDR = r"C:\Users\ofir\Documents\University\year2\GEC Project\GEC_ME_PROJECT-master\GEC_ME_PROJECT\DA\results\Batch_3727145_batch_results .csv"
MISTAKE_TYPES = ["Vt", "Vm", "V0", "Vform", "SVA", "ArtOrDet", "Nn", "Npos", "Pform", "Pref", "Prep", "Wci",
                 "Wa", "Wform", "Wtone", "Srun", "Smod", "Spar", "Sfrag", "Ssub", "WOinc", "WOadv", "Trans",
                 "Mec", "Rloc-", "Cit", "Others", "Um"]
EVALUATION_FEATURES = ["Nucle_ID"] + MISTAKE_TYPES + ["TotalMistakes", "z-score"]
TO_MTURK_FILE_PATH = r"C:\Users\ofir\Documents\University\year2\GEC Project\GEC_ME_PROJECT-master\GEC_ME_PROJECT\NUCLE\toMturk\mTurk_csv.csv"
SENTENCES_MISTAKES_SCORE = r"C:\Users\ofir\Documents\University\year2\GEC Project\GEC_ME_PROJECT-master\GEC_ME_PROJECT\DA\results\sentences_mistakes_scores.csv"


NUM_OF_BOOSTING_ITER = 10000
C = np.array([1, 3, 2, 4, 1, 2, -1, -3, 2, 1, 3, 4, 5, 1, -3, -6, 2, 1, 1, 2, 3, 1, -2, -5, 1, -2, 1, 4])



def get_senteces_by_df(nucle_addr,df,results_from_mturk_addr):
    lines = open(nucle_addr).read().splitlines()
    # hits = pd.read_csv(to_mturk_csv_addr)
    sentences = pd.DataFrame(columns= EVALUATION_FEATURES)
    # z_scores = get_all_z_scores(results_from_mturk_addr)
    for index, row in tqdm(df.iterrows()):
            sentences.loc[index] = [0]*len(EVALUATION_FEATURES)
            id = row["Input.Nucle_ID"] -1
            mistakes = parse_sentence(lines, id)[MISTAKES_INX]
            sentences.loc[index]["Nucle_ID"] = id
            for mistake in mistakes:
                sentences.loc[index][mistake[2]] += 1
                sentences.loc[index]["TotalMistakes"] += 1
            sentences.loc[index]["z-score"] = df.loc[index]["MistakeZScore"]
            # print(sentences.loc[index])
    sentences.to_csv("sentences_mistakes_scores.csv", sep=",", encoding='utf-8')
    return sentences


def mistakes_stats(sentences):
    sentences = get_X(sentences)
    df = pd.DataFrame(columns= ["number_of_sentences","perc_of_sentences", "total_appearance", "perc_of_appearance"])
    df["number_of_sentences"] = sentences.astype(bool).sum(axis=0)
    df["total_appearance"] = sentences.sum(axis=0)
    df["perc_of_sentences"] =  df["number_of_sentences"]/sentences.shape[0]
    df["perc_of_appearance"] =  df["total_appearance"]/df["total_appearance"]["TotalMistakes"]
    return df

def get_X(sentences):
    return sentences.drop('z-score',1).drop('Nucle_ID',1)

def get_weights(sentences, reg_flag):
    y = np.array(sentences["z-score"])
    if reg_flag:
        X = np.array(get_X(sentences))
        bias = 89138352537 #regression bug magik solution
    else:
        X = np.array(get_X(sentences).drop('TotalMistakes',1))
        bias = 0
    reg = LinearRegression().fit(X, y)
    return reg.coef_ - bias

def generate_y(xs):  # TODO: for DEBUG
    y=[0]*500
    for i in range(len(xs)):
        y[i] = C.dot(xs[i,:])
    return y


def boostrap_weights(sentences):
    mistakes = pd.DataFrame(columns=MISTAKE_TYPES)
    ranks = pd.DataFrame(columns=MISTAKE_TYPES)
    for i in tqdm(range(NUM_OF_BOOSTING_ITER)):
        sample = sentences.sample(n=sentences.shape[0], replace=True)
        weights = list(get_weights(sample, False))
        mistakes.loc["iter" + str(i)] = weights
        rank_row = [0] * len(weights)
        for j, x in enumerate(sorted(range(len(weights)), key=lambda y: weights[y])):
            rank_row[x] = j
        ranks.loc["iter" + str(i)] = rank_row
    uppers = []
    lowers = []
    for mis in mistakes:
        k = int(NUM_OF_BOOSTING_ITER*0.025)
        uppers.append(np.partition(mistakes[mis], -k)[-k])
        lowers.append(np.partition(mistakes[mis], -(NUM_OF_BOOSTING_ITER - k))[-(NUM_OF_BOOSTING_ITER - k)])

    mistakes.loc["upper"] = uppers
    mistakes.loc["lower"] = lowers
    mistakes.to_csv("debug.csv", sep=",", encoding='utf-8') #todo: for debug
    ranks.to_csv("ranks.csv", sep=",", encoding='utf-8') #todo: for debug
    return mistakes, mistakes.transpose()["upper"], mistakes.transpose()["lower"], ranks


def plot_mistakes(all_mistakes,mistakes,ranks):
    melted_mistakes = pd.melt(all_mistakes)
    melted_mistakes["weights"] = 0
    for mistake in MISTAKE_TYPES:
        val = mistakes["weights"][mistake]
        melted_mistakes["weights"][melted_mistakes["variable"] == mistake] = val


    melted_mistakes.sort_values(by=['weights'], inplace=True)
    sns.set(style="whitegrid")
    ax = sns.boxplot(x="variable", y="value", data=melted_mistakes)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    ax.set(xlabel='NUCLE error categories', ylabel='linear regression weight', title= 'NUCLE error categories sorted by "bothering" boxplots')
    plt.show()

    mistakes.sort_values(by=['weights'], inplace=True)
    mistakes.drop('TotalMistakes', inplace=True)
    # x = [mis for mis in mistakes.index if mis != "TotalMistakes"] #sorted labels
    means = np.array(mistakes["weights"])
    conf = np.array([[mistakes["lower"][mis], mistakes["upper"][mis]] for mis in mistakes.index])
    yerr = np.c_[means - conf[:, 0], conf[:, 1] - means].T
    print(yerr.T)
    plt.bar(mistakes.index, means, yerr=yerr, color="lightseagreen")
    plt.xlabel('NUCLE error categories')
    plt.ylabel('linear regression weight')
    plt.title('NUCLE error categories sorted by "bothering" with 95% CI')
    # plt.xticklabels(ax.get_xticklabels(),rotation=90)
    # plt.xticks([mis for mis in mistakes.index if mis != "TotalMistakes"])
    plt.xticks(rotation=90)
    plt.show()

    melted_ranks = pd.melt(ranks)
    melted_ranks["weights"] = 0
    for mistake in MISTAKE_TYPES:
        val = mistakes["weights"][mistake]
        melted_ranks["weights"][melted_ranks["variable"] == mistake] = val
    melted_ranks.sort_values(by=['weights'], inplace=True)
    bx = sns.barplot(x="variable", y="value", data=melted_ranks)
    bx.set_xticklabels(ax.get_xticklabels(), rotation=90)
    bx.set(xlabel='NUCLE error categories', ylabel="linear regression weight's rank", title='NUCLE error categories ranked weights (with CI) based on boosting 10K repeats')
    plt.show()



def analize(df):
    sentences = pd.read_csv(SENTENCES_MISTAKES_SCORE)
    sentences = sentences.loc[:, ~sentences.columns.str.contains('^Unnamed')]
    sentences = sentences.loc[~(sentences["TotalMistakes"] == 0),:]
    mistakes = mistakes_stats(sentences)
    mistakes["weights"] = list(get_weights(sentences, False)) + [0]
    mistakes["regularized_weights"] = get_weights(sentences, True)
    all_mistakes ,mistakes["upper"] , mistakes["lower"], ranks = boostrap_weights(sentences)
    mistakes.to_csv("mistakes_weights_with_regression.csv")
    plot_mistakes(all_mistakes, mistakes,ranks)

    return mistakes

def demo_analize():
    all_mistakes = pd.read_csv(r"C:\Users\ofir\Documents\University\year2\GEC Project\GEC_ME_PROJECT-master\GEC_ME_PROJECT\NUCLE\my_NUCLE_parser\debug.csv", index_col=0)
    mistakes = pd.read_csv(r"C:\Users\ofir\Documents\University\year2\GEC Project\GEC_ME_PROJECT-master\GEC_ME_PROJECT\NUCLE\my_NUCLE_parser\mistakes_weights_with_regression.csv", index_col=0)
    ranks = pd.read_csv(r"C:\Users\ofir\Documents\University\year2\GEC Project\GEC_ME_PROJECT-master\GEC_ME_PROJECT\NUCLE\my_NUCLE_parser\ranks.csv", index_col=0)
    plot_mistakes(all_mistakes, mistakes,ranks)


