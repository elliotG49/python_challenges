"""

Extract all vowels from a string using a list comprehension.

Create a list of numbers between 1–100 that are divisible by 7.

Given a list of words, get only the ones that start with 'p'.

From a list of tuples (name, age), extract names of people over 18.

"""

vowels = ["a", "e", "i", "o", "u"]
v = "hello please what is my name"
L = v.split(" ")
print(L)

words = [sorted(word, key=len) for word in v.split(" ")]

p = [char for char in v if char in vowels]
j = [char.upper() for char in v.split(" ") if len(v) > 29]
d = [num for num in range(1, 100) if num % 7 == 0]

x = ["Hello", "World", "Peers"]
l = [word for word in x if word[0].lower() == "p"]

people = [("Alice", 25), ("Bob", 10), ("Charlie", 22), ("Diana", 28), ("Ethan", 35)]
p = [person for person, age in people if age >= 18]


nums = [1, 2, 3, 5, 3, 2, 4, 3]
x = sorted(nums)
l = nums.sort()
print(len(v))
print(type(x), type(l), l, p, j)
x = lambda a: a * 2
print(x(2))


c = "hello"
l = "world"

print(c, end=" ")
print(l)

print("a" * -1 + "b" * 0 + "c" * 2)
print(words)
print([word for word in v.split(" ")])
