# filepath = '/home/elliotg/pythoning/py_learning/file_challenge/datafile.txt'
filepath = '/root/py_learning/file_challenge/datafile.txt'

def most_common_letter(filepath) -> dict:
    seen = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        data = f.read()
        for char in data:
            if char.isalpha():
                char = char.lower()
                seen[char] = seen.get(char, 0) + 1
        return seen
    
def sort_numbers(filepath) -> list:
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
        return sorted(numbers)
    
def count_words(filepath) -> int:
    words = 0
    with open(filepath, 'r') as f:
        filedata = f.read()
        for word in filedata.split(' '):
            if word.isalpha():
                words += 1
    return words

def parse_json(filepath) -> dict:
    import json
    read_line = False
    
    json_sig_start = '# JSON Data'
    json_sig_end = '# Unstructured Data'
    
    json_file = 'json_data.json'
    
    with open(filepath, 'r') as f:
        for line in f:
            if json_sig_end in line:
                read_line = False
            elif read_line:
                with open(json_file, 'a') as f:
                    f.write(line)
            elif json_sig_start in line:
                read_line = True
            else:
                continue
        with open(json_file) as f:
            json_data = json.load(f)
            return json_data
        
def main():
    # print(sort_numbers(filepath))
    print(most_common_letter(filepath))
    
    
main()