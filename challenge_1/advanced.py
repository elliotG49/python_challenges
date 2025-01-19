filepath = '/root/py_learning/file_challenge/datafile.txt'


def phone_number_formatter(filepath):
    import re

    with open(filepath, 'r') as f:
        data = f.read()
    matches = re.findall(r'\d{3}-\d{3}-\d{4}', data)

    for match in matches:
        formatted_number = f'+1-{match}'
        data = data.replace(match, formatted_number) 
    with open(filepath, 'w') as f:
        f.write(data)

    return data

        
def word_frequency(filepath) -> dict:
    seen = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        data = f.read()
        for word in data.split(' '):
            if word.isalpha():
                word = word.lower()
                seen[word] = seen.get(word, 0) + 1
        return seen
    
def obsfucate_email(filepath) -> list:
    import re
    sig = ['.', '@']
    obfuscated_emails = []
    with open(filepath, 'r') as f:
        data = f.read()
        email_match = re.findall(r'[\w\.-]+@[\w\.-]+', data)
        for email in email_match:
            # Construct the obfuscated email
            obfuscated_email = ''.join(
                f'[{char}]' if char in sig else char for char in email
            )
            obfuscated_emails.append(obfuscated_email)
        return obfuscated_emails
                
                

            
        
def main():
    print(phone_number_formatter(filepath))
    
main()