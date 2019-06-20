import pandas as pd
import numpy as np
from my_parser import NUCLE_DB_ADDR, parse_sentence, MISTAKES_INX

MISTAKE_TYPES = ["Vt", "Vm", "V0", "Vform", "SVA", "ArtOrDet", "Nn", "Npos", "Pform", "Pref", "Prep", "Wci",
                 "Wa", "Wform", "Wtone", "Srun", "Smod", "Spar", "Sfrag", "Ssub", "WOinc", "WOadv", "Trans",
                 "Mec", "Rloc-", "Cit", "Others", "Um"]
EVALUATION_FEATURES = ["Nucle_ID"] + MISTAKE_TYPES + ["TotalMistakes", "z-score"]


INPUT_FILE_PATH = r"C:\Users\ofir\Documents\University\year2\GEC Project\GEC_ME_PROJECT-master\GEC_ME_PROJECT\NUCLE\toMturk\mTurk_csv.csv"
lines = open(NUCLE_DB_ADDR).read().splitlines()
hits = pd.read_csv(INPUT_FILE_PATH)

sentences = pd.DataFrame(columns= EVALUATION_FEATURES)
for index, row in hits.iterrows():
    for i in range(1, 101):
        sentences.loc[i + 100*index] = [0]*len(EVALUATION_FEATURES)
        id = row["Nucle_ID" + str(i)] -1
        mistakes = parse_sentence(lines, id)[MISTAKES_INX]
        sentences.loc[i + 100*index]["Nucle_ID"] = id
        for mistake in mistakes:
            sentences.loc[i + 100 * index][mistake[2]] += 1
            sentences.loc[i + 100 * index]["TotalMistakes"] += 1

print(sentences)

if __name__ == '__main__':
    pass