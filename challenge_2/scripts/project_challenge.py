import pandas as pd
import numpy as np
import json



"""

1. Combine all learned skills to create a detailed report of employee-project assignments using data.json. The report should include:
2. Employee name and role.
3. Project name and department.
4. Save the report as a CSV file with the columns: Employee Name, Role, Project Name, Department.

"""



csv_filepath = '/root/py_learning/challenge_2/data/contacts.csv'
json_filepath = '/root/py_learning/challenge_2/data/data.json'
txt_filepath = '/root/py_learning/challenge_2/data/facts.txt'


def challenge_1():
    with open(json_filepath, 'r') as f:
        data = json.load(f)
        
        employees = data['employees']
        projects = data['projects']
        departments = data['departments']
        
        
        departments_df = pd.DataFrame(departments)
        projects_df = pd.DataFrame(projects)
        employees_df = pd.DataFrame(employees)
        
        df = pd.merge(departments_df, employees_df, left_on='department_id', right_on='id', how='left')
        
        df = pd.merge(df, projects_df, left_on='department_id', right_on='department_id', how='left')
        df.rename(columns={'name_x': 'department_name', 'name_y': 'employee_name', 'name': 'project_name', 'role': 'employee_role'}, inplace=True)
        df = df.drop(columns=['id'])
        df.loc[df['employee_name'] == 'Alice', 'employee_name'] = 'Jane'
        
        cols = list(df.columns)
        for i,col in enumerate(cols):
            print(f'Index: {i}, Column-Name: {col}')
        df = df[cols[2:4] + cols[0:2] + cols[4::]]
        managers = df.loc[df['employee_role'] == 'Manager']
        print(df, managers['employee_name'].tolist())

        
        
        
        
        
        
challenge_1()