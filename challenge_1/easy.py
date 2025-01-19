# filepath = '/home/elliotg/pythoning/py_learning/file_challenge/datafile.txt'
filepath = '/root/py_learning/file_challenge/datafile.txt'

    
def count_lines(file_data) -> int:
    file_lines = file_data.split("\n")
    counter = 0
    for i in file_lines:
        if file_lines:
            counter += 1
    return counter
            
def extratct_facts(filepath):
    signatures = ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', ] #im sure there is an easy way to do this, eg f'{digit}'.'
    with open(filepath, 'r') as file:
        for line in file:
            if line[:2] in signatures:
                print(line)
            else:
                continue

def sum_numbers(filepath):
    random_num_sig = '# Random Numbers'
    capture_next = False
    numbers = []
    with open(filepath, 'r') as f:
        for line in f:
            if capture_next:
                stripped_line = line.strip()
                for num in stripped_line.split(', '):
                    numbers.append(int(num))
            capture_next = False
            if random_num_sig in line:
                capture_next = True
        total = 0
        for num in numbers:
            total += num
            
        return total, numbers

def valid_emails(filepath):
    import re
    with open(filepath, 'r') as f:
        data = f.read()
        match = re.findall(r'[\w\.-]+@[\w\.-]+', data)
        for i in match:
            print(i) 
            

    

                
            
                
        
        
        
def main():
    # filedata = open_file(filepath)
    # number_of_lines = count_lines(filedata)
    # fact_lines = extratct_facts(filepath)
    print(sum_numbers(filepath))
    
main()