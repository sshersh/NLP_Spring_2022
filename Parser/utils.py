import string
from dataclasses import dataclass

@dataclass
class Node:
    """
    Class to store 'pointers' (indices) to each constituent in the table
    """
    l_val :  str
    rval_1 : list # should be a list of 3 indices
    rval_2 : list = None 

def make_grammar(input_file):
    """
    Take an input file and output the grammar as a list of rules
    """

    # store the terminal and non-terminal rules (POS) in separate dictionaries for convenience
    # in POS dictionary, keys are non-terminals and vals are POS
    non_terminal_rules = {}
    pos_rules = {}

    rules_file = open(input_file, 'r')
    rules_file_lines = rules_file.read().splitlines()
    
    for line_num, line in enumerate(rules_file_lines):
        split_line = line.split()
        if len(split_line) not in [3, 4]:
            exit("ERROR: Rule with invalid size detected on line " + str(line_num))

        l_symbol = split_line[0]   

        if len(split_line)  == 4:
            if non_terminal_rules.get(l_symbol) is None:
                non_terminal_rules[l_symbol] = [split_line[2:]]
            else:
                non_terminal_rules[l_symbol].append(split_line[2:])
        else:
            terminal = split_line[2]
            if pos_rules.get(terminal) is None:
                pos_rules[terminal] = [l_symbol]
            else:
                pos_rules[terminal].append(l_symbol)

    return non_terminal_rules, pos_rules