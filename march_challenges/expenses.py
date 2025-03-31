import pandas as pd
from datetime import date, datetime

def get_today_date():
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
    data = {}
    data_types = {
        "date": datetime,
        "account": str,
        "category": str,
        "note": str,
        "amount": float,
        "currency": str  # Changed from float to str for currency codes
    }

    for field, expected_type in data_types.items():
        while True:
            user_input = input(f"Enter {field} ({expected_type.__name__}): ")
            try:
                if expected_type == datetime:
                    converted_value = datetime.strptime(user_input, "%Y-%m-%d")  # Adjust format if needed
                elif expected_type == float:
                    converted_value = float(user_input)
                elif expected_type == int:
                    converted_value = int(user_input)
                elif expected_type == str:
                    converted_value = user_input.strip()  # Clean string input
                else:
                    raise ValueError("Unsupported data type")
                
                data[field] = converted_value
                break  # Exit loop if input is valid

            except ValueError:
                print(f"Invalid input for {field}. Expected {expected_type.__name__}. Try again.")

    # Convert to DataFrame and append to expense_file
    new_expense = pd.DataFrame([data])  # Convert dict to DataFrame
    expense_file = pd.concat([expense_file, new_expense], ignore_index=True)

    print("Expense added successfully!")
    return expense_file  # Return updated DataFrame
            
def main():
    path = '/home/ellio/Desktop/python_challenges/march_challenges/files/expenses.csv'
    file = open_csv(path)
    display_columns(file)
    get_total_expense(file)
    updated_file = add_new_expense(file)
    updated_file.to_csv(path, index=False)  # Overwrite CSV file with new data

    
main()
