from collections import Counter

types = Counter()
with open(r"/home/leshem/PycharmProjects/lab/GEC_ME_PROJECT/errant-master/sercl_auto_m2") as fl:
    for line in fl:
        if line.startswith("A"):
            types[line.split("|||")[1]] += 1
print(types)
print(types.most_common(70))
print(set(x[0] for x in types.most_common(70)))
