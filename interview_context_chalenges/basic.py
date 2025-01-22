import pandas as pd

data = {
    "IP_Address": ["192.168.0.1", "192.168.0.2", "10.0.0.1", "10.0.0.2", "172.16.0.1"],
    "Threat_Level": [3, 8, 5, 9, 2],
    "Timestamp": ["2025-01-20 12:34", "2025-01-21 08:45", "2025-01-20 14:12", "2025-01-21 09:34", "2025-01-20 10:00"],
    "Region": ["North America", "Europe", "Asia", "South America", "North America"]
}

def question_1a(data):
    df = pd.DataFrame(data)
    print(df.head(5))
    
def question_1b(data):
    df = pd.DataFrame(data)
    return df.isnull()

def question_1c(data):
    df = pd.DataFrame(data)
    x = df.loc[df['Threat_Level'] > 5]
    return x

def question_2a():
    from datetime import datetime
    with open('/root/py_learning/interview_context_chalenges/data/threat_data.csv', 'r') as f:
        df = pd.read_csv(f)
        for row in df['Timestamp']:
            row
            
        

question_2a()