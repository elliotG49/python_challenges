import pandas as pd

def open_csv(path):
    expense_file = pd.read_csv(path)
    return expense_file
    
def display_columns(expense_file):
    col = list(expense_file.columns)
    print(col)
    
def get_total_expense(expense_file):
    total = sum(expense_file['amount'])
    print(total)
    return total

def main():
    path = '/home/ellio/Desktop/python_challenges/march_challenges/files/expenses.csv'
    file = open_csv(path)
    display_columns(file)
    get_total_expense(file)

    
main()