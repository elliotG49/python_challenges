import numpy as np
import pandas as pd

"""

1. Text File Analysis: Count the total number of facts and quotes in facts.txt.
2. Extract Even Numbers: Extract all even numbers from the "Random Numbers" section of facts.txt.
3. Read CSV with Pandas: Load contacts.csv into a Pandas DataFrame and display the first 3 rows.

"""

csv_filepath = '/root/py_learning/challenge_2/data/contacts.csv'
json_filepath = '/home/elliotg/pythoning/py_learning/challenge_2/data/data.json'
txt_filepath = '/root/py_learning/challenge_2/data/facts.txt'

def challenge_1():
    import re
    with open(txt_filepath, 'r') as f:
        fact_pattern = r'^\d+\.\s+(.*)'
        quote_pattern = r'^"(.*?)"\s*-\s*(.*)'
        
        data  = f.read()
        
        facts = re.findall(fact_pattern, data, re.MULTILINE)
        quotes = re.findall(quote_pattern, data, re.MULTILINE)
        
        number_of_facts = len(facts)
        number_of_quotes = len(quotes)
        
        return number_of_facts + number_of_quotes, data


def challenge_2():
    string_start = '# Random Numbers'
    string_end = '\n\n'
    flag = False
    even_number = []
    with open(txt_filepath, 'r') as f:
        for line in f:
            if string_end in line:
                flag = False
            elif flag:
                for char in line:
                    if char.isdigit():
                        if int(char) % 2 == 0:
                            even_number.append(char)
            elif string_start in line:
                flag = True
            else:
                continue
        return set(even_number)
    
def challenge_3():
    df = pd.read_csv(csv_filepath)
    return df.head(3)



                
                
            
        
