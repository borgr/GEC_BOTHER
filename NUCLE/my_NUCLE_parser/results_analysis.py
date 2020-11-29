import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from my_parser import NUCLE_DB_ADDR, parse_sentence, MISTAKES_INX

### FILES ###
RESULTS_FILE_ADDR = r"DA/results/Batch_3727145_batch_results.csv"
TO_MTURK_FILE_PATH = r"NUCLE/toMturk/mTurk_csv.csv"
SENTENCES_MISTAKES_SCORE = r"DA/results/sentences_mistakes_scores.csv"
ERRANT_SENTENCES_MISTAKES_SCORE = r"DA/results/sentences_mistakes_scores_errant.csv"
ERRANT_DB_ADDR = r"errant-master/out_auto_m2"

# featurs:
NUCLE_MISTAKE_TYPES = ["Vt", "Vm", "V0", "Vform", "SVA", "ArtOrDet", "Nn", "Npos", "Pform", "Pref", "Prep",
                       "Wci",
                       "Wa", "Wform", "Wtone", "Srun", "Smod", "Spar", "Sfrag", "Ssub", "WOinc", "WOadv",
                       "Trans",
                       "Mec", "Rloc-", "Cit", "Others", "Um"]

ERRANT_MISTAKE_TYPES = ['R:VERB:FORM', 'R:VERB:TENSE', 'M:NOUN', 'U:PREP', 'M:VERB:TENSE', 'R:PRON',
                        'R:SPELL',
                        'R:ADV', 'U:NOUN:POSS', 'R:ORTH', 'M:DET', 'U:OTHER', 'R:NOUN', 'U:ADV', 'M:ADJ',
                        'R:PREP',
                        'R:CONJ', 'U:NOUN', 'M:VERB:FORM', 'R:NOUN:NUM', 'R:NOUN:POSS', 'U:VERB:FORM',
                        'R:CONTR',
                        'M:ADV', 'U:DET', 'U:PUNCT', 'M:PRON', 'U:PART', 'R:NOUN:INFL', 'M:PART', 'M:CONJ',
                        'R:VERB:SVA',
                        'M:PUNCT', 'R:PUNCT', 'M:NOUN:POSS', 'U:VERB:TENSE', 'U:PRON', 'U:CONJ', 'R:WO',
                        'R:VERB',
                        'R:ADJ', 'M:PREP', 'R:VERB:INFL', 'U:ADJ', 'U:VERB', 'M:OTHER', 'U:CONTR', 'UNK',
                        'R:OTHER',
                        'R:PART', 'R:MORPH', 'R:DET', 'M:VERB', 'R:ADJ:FORM']

EVALUATION_FEATURES = ["Nucle_ID"] + NUCLE_MISTAKE_TYPES + ["TotalMistakes", "z-score"]
NUM_OF_BOOTSTRAP_ITER = 10000


def nucle_to_errant_id_dict(nucle_addr, errant_addr):
    """
    create a dictionary of id's  nucle of form- original line : errant line
    :param nucle_addr: original NUCLE m2 file addr
    :param errant_addr: ERRANT m2 file addr
    :return:
    """
    with open(nucle_addr) as fl:
        nucle_lines = fl.read().splitlines()
    with open(errant_addr, encoding="ISO-8859-2") as fl:
        errant_lines = fl.read().splitlines()
    n_counter, e_counter, n_len = 0, 0, len(nucle_lines)
    d = dict()
    while True:
        if n_counter >= n_len:
            break
        if nucle_lines[n_counter].startswith("S"):
            while not errant_lines[e_counter].startswith("S"):
                e_counter += 1
            d[n_counter] = e_counter
            e_counter += 1
            print(n_counter)
        n_counter += 1
    return d


def get_ERRANT_atributes():
    """
    :return: all posible mistakes by errant
    """
    errant_lines = open(ERRANT_DB_ADDR).read().splitlines()
    a = set()
    for id, line in enumerate(errant_lines):
        if line.startswith("S"):
            mistakes = parse_sentence(errant_lines, id)[MISTAKES_INX]
            for mistake in mistakes:
                a.add(mistake[2])
    return a


def get_senteces_by_df(nucle_addr, df, errant):
    """
    :param nucle_addr: original m2 NUCLE file
    :param df: melted filtered df
    :param errant: if true, use errant id, else use nucle id.
    :return: df in which every sentence is a vector of mistakes with z-score
    """
    if errant:
        global EVALUATION_FEATURES
        EVALUATION_FEATURES = ["Nucle_ID"] + ERRANT_MISTAKE_TYPES + ["TotalMistakes", "z-score"]
    with open(nucle_addr, encoding="ISO-8859-2") as fl:
        lines = fl.read().splitlines()
    sentences = pd.DataFrame(columns=EVALUATION_FEATURES)
    d = nucle_to_errant_id_dict(NUCLE_DB_ADDR, ERRANT_DB_ADDR)
    for index, row in tqdm(df.iterrows()):
        sentences.loc[index] = [0] * len(EVALUATION_FEATURES)
        id = row["Input.Nucle_ID"] - 1
        if errant:
            id = d[id]
        mistakes = parse_sentence(lines, id)[MISTAKES_INX]
        sentences.loc[index]["Nucle_ID"] = id
        for mistake in mistakes:
            sentences.loc[index][mistake[2]] += 1
            sentences.loc[index]["TotalMistakes"] += 1
        sentences.loc[index]["z-score"] = df.loc[index]["MistakeZScore"]
    sentences.to_csv("sentences_mistakes_scores_errant.csv", sep=",", encoding='utf-8')
    return sentences


def mistakes_stats(sentences):
    """
    create a df of statistics of the different mistakes
    :param sentences: df in which every sentence is a vector of mistakes with z-score
    :return: df of statistics of the different mistakes
    """
    sentences = get_X(sentences)
    df = pd.DataFrame(
        columns=["number_of_sentences", "perc_of_sentences", "total_appearance", "perc_of_appearance"])
    df["number_of_sentences"] = sentences.astype(bool).sum(axis=0)
    df["total_appearance"] = sentences.sum(axis=0)
    df["perc_of_sentences"] = df["number_of_sentences"] / sentences.shape[0]
    df["perc_of_appearance"] = df["total_appearance"] / df["total_appearance"]["TotalMistakes"]
    return df


def get_X(sentences):
    """
    get the domain for the regression - clean irrelevant columns from the df
    :param sentences: senteces df
    :return: same df without irrelevant columns
    """
    return sentences.drop('z-score', 1).drop('Nucle_ID', 1)


def get_weights(sentences, reg_flag):
    """
    perforn linear regression and return the weights
    :param sentences: df of sentences as a vector of mistakes and a z-score
    :param reg_flag: if Ture - include number of mistakes in the regression, else - don't include it
    :return: regression weights a a vector.
    """
    y = np.array(sentences["z-score"])
    if reg_flag:
        X = np.array(get_X(sentences))
        bias = 89138352537  # regression bug magic solution, all results had this bias, may change on different computer
    else:
        X = np.array(get_X(sentences).drop('TotalMistakes', 1))
        bias = 0
    reg = LinearRegression().fit(X, y)
    return reg.coef_ - bias


def bootstrap_weights(sentences, errant):
    """
    resample and estimate data values NUM_OF_BOOTSTRAP_ITER times.
    :param sentences: df of sentences as a vector of mistakes and a z-score
    :param errant: if true, use errant id, else use nucle id.
    :return:
    mistakes - a df of mistake stats
    mistakes.transpose()["upper"] - a vector of top 97.5% scores for the different mistake types
    mistakes.transpose()["lower"] - a vector of bottom 2.5% scores for the different mistake types
    ranks - a df of mistake's ranks over the iterations
    """
    if errant:
        types = ERRANT_MISTAKE_TYPES
    else:
        types = NUCLE_MISTAKE_TYPES
    mistakes = pd.DataFrame(columns=types)
    ranks = pd.DataFrame(columns=types)
    for i in tqdm(range(NUM_OF_BOOTSTRAP_ITER)):
        sample = sentences.sample(n=sentences.shape[0], replace=True)
        weights = list(get_weights(sample, False))
        mistakes.loc["iter" + str(i)] = weights
        rank_row = [0] * len(weights)
        for j, x in enumerate(sorted(range(len(weights)), key=lambda y: weights[y])):
            rank_row[x] = j
        ranks.loc["iter" + str(i)] = rank_row
    uppers = []
    upper_ranks = []
    lowers = []
    lower_ranks = []
    for mis in mistakes:
        k = int(NUM_OF_BOOTSTRAP_ITER * 0.025)
        uppers.append(np.partition(mistakes[mis], -k)[-k])
        lowers.append(np.partition(mistakes[mis], -(NUM_OF_BOOTSTRAP_ITER - k))[-(NUM_OF_BOOTSTRAP_ITER - k)])
        upper_ranks.append(np.partition(ranks[mis], -k)[-k])
        lower_ranks.append(
            np.partition(ranks[mis], -(NUM_OF_BOOTSTRAP_ITER - k))[-(NUM_OF_BOOTSTRAP_ITER - k)])

    mistakes.loc["upper"] = uppers
    mistakes.loc["lower"] = lowers
    mistakes.loc["lower_ranks"] = lower_ranks
    mistakes.loc["upper_ranks"] = upper_ranks
    mistakes.loc["mean_rank"] = [np.mean(ranks[mt]) for mt in ranks]
    mistakes.to_csv("debug_errant.csv", sep=",", encoding='utf-8')  # save for debug
    ranks.to_csv("ranks_errant.csv", sep=",", encoding='utf-8')  # save for debug
    return mistakes, ranks


def plot_mistakes(all_mistakes, mistakes, ranks, errant):
    """
    plot graph of mistakes scores
    :param all_mistakes: a df with all bootstrap repeats scores
    :param mistakes: mistakes statistics
    :param ranks: mistake ranks of the different bootstrapping iterations df
    :param errant: if true, use errant id, else use nucle id.
    :return: None. plot 3 graphs.
    """
    if errant:
        types = ERRANT_MISTAKE_TYPES
    else:
        types = NUCLE_MISTAKE_TYPES

    melted_mistakes = pd.melt(all_mistakes)
    melted_mistakes["weights"] = 0
    for mistake in types:
        # if mistake != "TotalMistakes":
        val = mistakes["weights"][mistake]
        melted_mistakes["weights"][melted_mistakes["variable"] == mistake] = val

    # graph 1 - boxplots:
    melted_mistakes.sort_values(by=['weights'], inplace=True)
    sns.set(style="whitegrid")
    ax = sns.boxplot(x="variable", y="value", data=melted_mistakes)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set(xlabel='error categories', ylabel='linear regression weight',
           title='ERRANT error categories sorted by "bothering" boxplots')
    plt.show()

    # graph 2 - bars:
    mistakes.sort_values(by=['weights'], inplace=True)
    mistakes.drop('TotalMistakes', inplace=True)
    # x = [mis for mis in mistakes.index if mis != "TotalMistakes"] #sorted labels
    means = np.array(mistakes["weights"])
    conf = np.array([[mistakes["lower"][mis], mistakes["upper"][mis]] for mis in mistakes.index])
    yerr = np.c_[means - conf[:, 0], conf[:, 1] - means].T
    plt.bar(mistakes.index, means, yerr=yerr, color="lightseagreen")
    plt.xlabel('error categories')
    plt.ylabel('linear regression weight')
    plt.title('ERRANT error categories sorted by "bothering" with estimated 95% CI based on boostrapping')
    plt.xticks(rotation=90)
    plt.show()

    # graph 3 - ranks:
    mistakes.sort_values(by=['weights'], inplace=True)
    # x = [mis for mis in mistakes.index if mis != "TotalMistakes"] #sorted labels
    means = np.array(mistakes["mean_rank"])
    conf = np.array([[mistakes["lower_rank"][mis], mistakes["upper_rank"][mis]] for mis in mistakes.index])
    yerr = np.c_[means - conf[:, 0], conf[:, 1] - means].T
    plt.bar(mistakes.index, means, yerr=yerr, color="darksalmon")
    plt.xlabel('error categories')
    plt.ylabel("linear regression weight's rank")
    plt.title('ERRANT error categories average rank (with 95% CI) based on bootstrapping 10K repeats')
    plt.xticks(rotation=90)
    plt.show()

    # # graph 4 - ranks:
    # melted_ranks = pd.melt(ranks)
    # melted_ranks["weights"] = 0
    # for mistake in NUCLE_MISTAKE_TYPES:
    #     val = mistakes["weights"][mistake]
    #     melted_ranks["weights"][melted_ranks["variable"] == mistake] = val
    # melted_ranks.sort_values(by=['weights'], inplace=True)
    # # print("melted ranks: \n",melted_ranks.head(50))
    # bx = sns.barplot(x="variable", y="value", data=melted_ranks)
    # bx.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # bx.set(xlabel='ERRANT error categories', ylabel="linear regression weight's rank",
    #        title='ERRANT NUCLE error categories ranked weights (with CI) based on bootstrapping 10K repeats')
    # plt.show()


def analyze(df, errant):
    """
    gets df of senteces as vectors and analize them - save stats and plot graphs
    :param df:  pandas df of sentences as a vector of mistakes and a z-score. 0 if using file.
    :param errant: if true, use errant id, else use nucle id.
    :return: mistakes stats
    """
    if df is not None:
        sentences = get_senteces_by_df(ERRANT_DB_ADDR, df,
                                       True)  # if changes filter use this instead of loading from file
    else:
        sentences = pd.read_csv(
            ERRANT_SENTENCES_MISTAKES_SCORE)  # can be used on ERRANT_SENTENCES_MISTAKES_SCORE or SENTENCES_MISTAKES_SCORE
        sentences = sentences.loc[:, ~sentences.columns.str.contains('^Unnamed')]
    sentences = sentences.loc[~(sentences["TotalMistakes"] == 0), :]
    mistakes = mistakes_stats(sentences)
    mistakes["weights"] = list(get_weights(sentences, False)) + [0]
    mistakes["regularized_weights"] = get_weights(sentences, True)
    all_mistakes, ranks = bootstrap_weights(sentences, errant)
    mistakes["upper"], mistakes["lower"], mistakes["upper_rank"], mistakes["lower_rank"], = \
        all_mistakes.transpose()["upper"], all_mistakes.transpose()["lower"], all_mistakes.transpose()[
            "upper_ranks"], all_mistakes.transpose()["lower_ranks"]
    mistakes["mean_rank"] = all_mistakes.transpose()["mean_rank"]
    mistakes.to_csv("mistakes_weights_with_regression_errant.csv", sep=",", encoding='utf-8')
    all_mistakes.drop(["upper", "lower", "lower_ranks", "upper_ranks", "mean_rank"], inplace=True)
    plot_mistakes(all_mistakes, mistakes, ranks, errant)
    return mistakes


def demo_analyze():
    """
    anlyze-like function, loads the bootstrapped data from file instead of re-creating it.
    :return:
    """
    all_mistakes = pd.read_csv(
        r"NUCLE\my_NUCLE_parser\debug.csv",
        index_col=0)
    mistakes = pd.read_csv(
        r"NUCLE\my_NUCLE_parser\mistakes_weights_with_regression.csv",
        index_col=0)
    ranks = pd.read_csv(
        r"NUCLE\my_NUCLE_parser\ranks.csv",
        index_col=0)
    plot_mistakes(all_mistakes, mistakes, ranks, False)
