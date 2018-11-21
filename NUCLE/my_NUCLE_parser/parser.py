# a NUCLE parser for ME (mistake evaluation) project.

import sys;


ATRIBUTES = "A"
SENTENCE = "S"



def get_sentance(source_file):
    while :
        line = source_file.readline()
        while line[0] == ATRIBUTES:




def main():
    file_name = sys.argv[1] #NUCLE db address
    with open(file_name, 'r') as file:
        cur_sentance = get_sentance(file)





if __name__ == '__main__':
    main();