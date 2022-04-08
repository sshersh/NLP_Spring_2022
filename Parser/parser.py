import string
from utils import *
from copy import deepcopy

def cky(sentence, NT_rules, pos_rules):
    """
    Main function which parses the sentence and outputs a list of valid parses

    sentence            :   string that is to be parsed
    NT_rules  :   dict of NT rules in form {A : [B C]}
    pos_rules           :   dict of POS rules in form {x : A} where x is terminal
    """

    words = sentence.split()
    sentence_len = len(words)

    parse_table = [[[] for i in range(j)] for j in range(1, sentence_len + 1)]
    parse_table_deriv = [[[] for i in range(j)] for j in range(1, sentence_len + 1)]

    for j, word in enumerate(words):
        # first find parts of speech for major diagonal
        pos_list = pos_rules.get(word)
        parse_table[j][j] = pos_list
        parse_table_deriv[j][j] = [Node(pos, [j, j, ind]) for ind, pos in enumerate(pos_list)]

        # now iterate through rest of column
        for i in range(j - 1, -1, -1):
            for k in range(i , j):
                for A in NT_rules:
                    for rule in NT_rules[A]:
                        if rule[0] in parse_table[k][i] \
                            and rule[1] in parse_table[j][k + 1]:
                            parse_table[j][i].append(A)
                            ind1 = parse_table[k][i].index(rule[0])
                            ind2 = parse_table[j][k + 1].index(rule[1])
                            parse_table_deriv[j][i].append(Node(A, [k, i, ind1], [j, k + 1, ind2]))

    return parse_table, parse_table_deriv

def print_parses_toplevel(sentence, parse_table, parse_table_deriv, node):
    print_parses(sentence, parse_table, parse_table_deriv, node)
    print('')

def print_parses(sentence, parse_table, parse_table_deriv, node):
    """
    Recursively print the parses in bracketed notation
    """
    words = sentence.split()
    print('[' + node.l_val + ' ', end = '')
    if node.rval_2 is None:
        word = words[node.rval_1[0]]
        print(word + ']', end = '')
    else:
        node2 = parse_table_deriv[node.rval_1[0]][node.rval_1[1]][node.rval_1[2]]
        node3 = parse_table_deriv[node.rval_2[0]][node.rval_2[1]][node.rval_2[2]]
        print_parses(sentence, parse_table, parse_table_deriv, node2)
        print_parses(sentence, parse_table, parse_table_deriv, node3)
        print(']', end = '')

sentence = 'i book the flight to houston'
NT_rules, pos_rules = make_grammar('sampleGrammar.cnf')
parse_table, parse_table_deriv =cky(sentence, NT_rules, pos_rules)

print_parses_toplevel(sentence, parse_table, parse_table_deriv, parse_table_deriv[-1][0][0])
