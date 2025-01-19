import pandas as pd
import numpy as np
import re
import json


"""

1. Count Words in Quotes: Count the total number of words across all quotes in facts.txt.
2. Merge DataFrames: Create two DataFrames from data.json (employees and departments) and merge them on department_id.
3. Extract Unique Domains: Extract and count unique email domains from contacts.csv.
4. JSON to DataFrame: Convert the projects section of data.json into a Pandas DataFrame and save it as a CSV file.
5. Most Common Word: Find the most common word in facts.txt (ignoring case and punctuation).


"""

csv_filepath = '/root/py_learning/challenge_2/data/contacts.csv'
json_filepath = '/root/py_learning/challenge_2/data/data.json'
txt_filepath = '/root/py_learning/challenge_2/data/facts.txt'

def challenge_1():
    sig = r'"([^"]+)"'
    total_words = 0
    with open(txt_filepath, 'r') as f:
        data = f.read()
        quotes = re.findall(sig, data, re.MULTILINE)
        for quote in quotes:
            words = quote.split(' ')
            for word in words:
                total_words += 1
                
        return total_words
                
def challenge_2():
    with open(json_filepath, 'r') as f:
        json_file = json.load(f)
        
        departments = json_file['departments']
        employees = json_file['projects']
        
        dp_df = pd.DataFrame(departments)
        emp_df = pd.DataFrame(employees)
        
        df = pd.merge(dp_df, emp_df, on="department_id")
        print(df)

def challenge_3():
    domain_sig = r'@[a-z]*.[a-z]*'
    df = pd.read_csv(csv_filepath)
    emails = df['Email']
    for email in emails:
        domains = re.findall(domain_sig, email)
    print(domains)
    
def challenge_4():
    with open(json_filepath, 'r') as f:
        json_obj = json.load(f)
        projects_dict = json_obj['projects']
        projects_df = pd.DataFrame(projects_dict)
        projects_df.to_csv('challenge_4.csv', index=False)
        

def challenge_5():
    import string
    seen = {}
    exclude = set(string.punctuation)
    with open(txt_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            words = line.split(' ')
            for word in words:
                if word.isalpha():
                    word = ''.join(ch for ch in word if ch not in exclude)
                    if word in seen:
                        seen[word] += 1
                    else:
                        seen[word] = 1
                else:
                    continue
        sorted_dict = dict(sorted(seen.items(), key=lambda item: item[1], reverse=True))
        first_value = next(iter(sorted_dict.keys()))
        return first_value
        
challenge_4()


