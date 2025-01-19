import pandas as pd
import numpy as np
import re

"""

1. Find Longest Quote: Identify the longest quote in facts.txt and print its length.
2. Filter Contacts: Using Pandas, filter contacts in contacts.csv to include only those with a phone number starting with 555.
3. Group Data (Pandas): Load data.json into a Pandas DataFrame and group projects by their department_id.
4. Sum Random Numbers: Calculate the sum of numbers in the "Random Numbers" section of facts.txt.
5. Update CSV Data: Add a new column to contacts.csv called Location and fill it with "Unknown". Save the updated file.

"""

csv_filepath = '/root/py_learning/challenge_2/data/contacts.csv'
json_filepath = '/root/py_learning/challenge_2/data/data.json'
txt_filepath = '/root/py_learning/challenge_2/data/facts.txt'


def challenge_1():
    string_start = '# Random Facts'
    string_end = '# Quotes'
    flag = False
    length = 0
    with open(txt_filepath, 'r') as f:
        for line in f:
            if string_end in line:
                flag = False
            elif flag:
                if len(line) > length:
                    length = len(line)
                    
            elif string_start in line:
                flag = True
            else:
                continue
        return length

def challenge_2():
    df = pd.read_csv(csv_filepath)
    phone_filter = df.loc[df['Phone'].str.contains('^555', regex=True)]
    return phone_filter

def challenge_3():
    df = pd.read_json(json_filepath)
    projects_df = pd.json_normalize(df['projects'])
    departments_df = pd.json_normalize(df['departments'])
    grouped_by_id = projects_df.groupby('department_id')
    for dept_id, group in grouped_by_id:
        print(f"\nDepartment ID: {dept_id}")
        print(group[['project_id', 'name']])
    
def challenge_4(txt_filepath):
    string_start = '# Random Numbers'
    flag = False
    numbers = []

    with open(txt_filepath, 'r') as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            if flag:
                parts = line.split(',')
                for part in parts:
                    part = part.strip()  # Remove extra spaces
                    if part.isdigit():
                        numbers.append(part)
            elif string_start in line:
                flag = True  # Start processing after the '# Random Numbers' line
        total = 0
        for num in numbers:
            total += int(num)
        return total

def challenge_5():
    df = pd.read_csv(csv_filepath)
    df['Location']  = 'Unknown'
    df.to_csv('modified_contacts.csv')
    
    
def main():
    challenge_5()
    
main()