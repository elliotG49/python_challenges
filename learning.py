import string

def count_vowels(string) -> int:
    vowels = 'aeioui'
    vowel_count = 0
    for char in string:
        if char in vowels:
            vowel_count += 1
    return vowel_count

def prime_number(integer) -> bool:
    prime = True
    for i in range(2, integer-1):
        if integer % i == 0:
            prime = False
            return prime, i
    return prime
        

def reverse_string_without_slicing(string) -> str:
    new_string = []
    string  = list(string)
    for i in range(len(string)):
        print(string)
        new_string.append(string[-1])
        string.pop()

    return new_string



def rock_paper_scissors() -> str:
    import random
    answers = ['rock', 'scissors', 'paper']
    user_input = str(input('Enter: ')).strip()
    challenge = random.choice(answers)
    if user_input not in answers:
        print('incorrect input')
        
def count_words(sentence):
    new_sentence = sentence.split(' ')
    print(len(new_sentence))

def sum_of_digits(integer) -> int:
    return sum([int(d) for d in str(integer)])
    

def largest_number(n_array) -> int:
    largest_n = 0
    for num in n_array:
        if num > largest_n:
            largest_n = num
        else:
            continue
    return largest_n

def remove_duplicates(array):
    return set(array)

def remove_duplicates_without_set(array):
    seen = []
    for item in array:
        if item in seen:
            continue
        else:
            seen.append(item)
        
    return seen


def find_missing_number(array) -> int:
    last_number = []
    for item in array:
        if not last_number:
            last_number.append(item)
        elif item == last_number[0] + 1:
            last_number.pop()
            last_number.append(item)
        else:
            missing_n = (last_number[0] + 1)
            return missing_n

            
print(find_missing_number([1, 2, 3, 4, 5, 7]))

    
    
    