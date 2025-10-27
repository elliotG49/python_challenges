from random import randint as rand
from dataclasses import dataclass


def make_coordinates() -> tuple[int, int]:
    x = rand(0, 100)
    y = rand(0, 100)
    return (x, y)


def colors() -> list[str]:
    colors = ["red", "blue", "yellow", "orange", "black", "white"]
    r = rand(0, len(colors) - 1)
    return [colors[r]]


def vehicle_type() -> list[str]:
    cars = ["Peugot", "Ford", "Mazda", "BMW", "Mercedes", "Nissan"]
    r = rand(0, len(cars) - 1)
    return [cars[r]]


class Car:
    def __init__(self) -> None:
        print("Car created")

    def __del__(self) -> None:
        print("Car removed")


@dataclass
class Coordinates:
    x: int
    y: int


cords = []
for _ in range(5):
    c = Coordinates(*make_coordinates())
    cords.append(c)

p = [c.x for c in cords]
print(p)


def add(*args):
    m = max(args)
    return m if m > 1000 else min(args)


def show(**kwargs):
    print(kwargs)


word1 = "hello"
word2 = "world"


def xor_words(word1: str, word2: str) -> list[int]:
    """_summary_

    Args:
        word1 (str): _description_
        word2 (str): _description_

    Returns:
        list[int]: _description_
    """
    xored = []
    w1_ord = [ord(char) for char in word1]
    w2_ord = [ord(char) for char in word2]

    print(w1_ord)
    print(w2_ord)

    for i, _ in enumerate(min(w1_ord, w2_ord)):
        xored.append(w1_ord[i] ^ w2_ord[i])
    return xored


print(xor_words(word1, word2))


tuple = ("Hello", 1, 8.8)
sets = set()

print(type(tuple))
print(type(sets))
