def two_length_list():
    """
    
    Have a variable set for current 
    
    """
    
    ch_counter = 0
    
    words = ["flower","fight"]
    prefix = ""
    for char in words[1]:
        if char != words[0][ch_counter]:
            return prefix
        else:
            prefix += char
            ch_counter += 1
    return prefix


def longest_common_prefix(strings):
    if not strings:
        return ""  # Handle edge case of an empty array

    # Start with the first string as the prefix
    prefix = strings[0]

    # Compare the prefix with every other string
    for string in strings[1:]:
        while not string.startswith(prefix):
            # Shorten the prefix until it matches the start of the string
            prefix = prefix[:-1]
            if prefix == "":
                return ""  # No common prefix

    return prefix

        
def longest_common_prefix(strings):
    if not strings:
        return ""

    for i in range(len(strings[0])):
        char = strings[0][i]  # Take the character from the first string
        for string in strings[1:]:
            # If index i is out of bounds or characters don't match
            if i >= len(string) or string[i] != char:
                return strings[0][:i]  # Return prefix up to this point

    return strings[0]  # All strings share the entire first string as a prefix

def valid_parenthesis(string):
    """
    Determine if the input string is valid:
    - Open brackets must be closed by the same type of brackets.
    - Open brackets must be closed in the correct order.
    """
    valid_correspondent = {"(": ")", "[": "]", "{": "}"}
    stack = []

    for char in string:
        if char in valid_correspondent:  # If it's an opening bracket
            stack.append(char)  # Push to stack
        elif char in valid_correspondent.values():  # If it's a closing bracket
            if stack and valid_correspondent[stack[-1]] == char:
                stack.pop()  # Pop the matching opening bracket
            else:
                return False  # Invalid if no match or stack is empty

    return len(stack) == 0  # Valid if no unmatched opening brackets remain

def merge_sorted_lists(list1, list2):
    list1 = sorted(list1)  # Ensure lists are sorted
    list2 = sorted(list2)

    sorted_list = []
    i, j = 0, 0  # Pointers for list1 and list2

    # Merge lists until one is exhausted
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            print(f'added {list1[i]} as it is smaller or equal to {list2[j]}')
            sorted_list.append(list1[i])
            i += 1
        else:
            print(f'added {list2[j]} as it is smaller than {list1[i]}')
            sorted_list.append(list2[j])
            j += 1

    # Add any remaining elements from list1
    while i < len(list1):
        print(f'added remaining {list1[i]}')
        sorted_list.append(list1[i])
        i += 1

    # Add any remaining elements from list2
    while j < len(list2):
        print(f'added remaining {list2[j]}')
        sorted_list.append(list2[j])
        j += 1

    print(sorted_list)
    return sorted_list



def remove_element(int_arr, val) -> int:
    """
    
    Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The order of the elements may be changed. Then return the number of elements in nums which are not equal to val.

    Consider the number of elements in nums which are not equal to val be k, to get accepted, you need to do the following things:

    Change the array nums such that the first k elements of nums contain the elements which are not equal to val. The remaining elements of nums are not important as well as the size of nums.
    Return k.
    
    """
    
    int_length = len(int_arr)
    counter = 0
    val_counter = 0
    while counter < int_length:
        if int_arr[counter] == val:
            int_arr.remove(int_arr[counter])
            int_arr.append('_')
            counter -= 1
            val_counter += 1
            
        else:
            counter += 1
    k = int_length - val_counter
    return int_arr, k
    
    
    
def index_in_string(haystack, needle):
    """
    
    Given two strings needle and haystack, return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

 
    """
    pointer = 0
    needle_length = len(needle)
    while pointer + needle_length <= len(haystack):
        section = haystack[pointer:pointer+3]
        if needle in section:
            return pointer
        else:
            pointer += 1
    return -1
    
        
    
def search_insert_pos(arr, target):
    """
    
    Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

    You must write an algorithm with O(log n) runtime complexity.
    
    """
    counter = 0
    while counter < len(arr):
        if arr[counter] == target:
            return counter
        elif arr[counter] > target:
            return counter
        else:
            counter += 1
    return len(arr)



def length_of_last_word(string):
    """
    
    Given a string s consisting of words and spaces, return the length of the last word in the string.

    A word is a maximal 
    substring
    consisting of non-space characters only.
    
    """
    pointer = -1
    value = string[pointer]
    while not value.isalpha():
        pointer -= 1
        
    return value, pointer
    
    
    
    
def increment(int_arr):
    """
    
    You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading 0's.

    Increment the large integer by one and return the resulting array of digits.
    
    """

    last_val = int_arr[-1]
    if last_val != 9:
        int_arr[-1] += 1
    else:
        int_arr[-1] = 1
        int_arr.append(0)
    return int_arr

arr = [9]
print(increment(arr))



    
    
    


         
    
            
    
    
    
