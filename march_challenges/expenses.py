import pandas as pd

def get_today_date():
    from datetime import date
    today = date.today()
    return today

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

def add_new_expense(expense_file):
    data = []
    data_types = {"date": datetime ,"account": ["Personal", "Business"], "catagory": str ,"note": str,"amount": float, "currency": float}
    while input_val != limit:
        for type in data_types:
            user_input = input(f"Enter {type}: ")
            if user_input == data_types[type].value():
                data.append(input)
                input_val += 1
    expense_file.loc(data)
            
def main():
    path = '/home/elliotg/Desktop/python_challenges/march_challenges/files/expenses.csv'
    file = open_csv(path)
    display_columns(file)
    get_total_expense(file)

    
main()
