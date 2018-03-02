from collections import deque,Counter
from bisect import insort, bisect_left
from itertools import islice
from __future__ import absolute_import, division, print_function
from __future__ import division
import random
from Queue import PriorityQueue
import sys

=======================================      ====================================
Hackerank solved python
https://www.martinkysel.com/hackerrank-make-it-anagram-solution/
https://github.com/gauravprasad/CodingChallenge/tree/master/HackerRank/src/main/java/com/gprasad/hackerrank/tutorial/crackingcodinginterview/ds


5 4
1 2 3 4 5

n, d = map(int, raw_input().strip().split(' '))
a = map(int, raw_input().strip().split(' '))

print ' '.join(map(str,answer))  # answer = [] i.e expecting an array


4
add hack
add hackerrank
find hac
find hak

n = int(raw_input().strip())
for a0 in xrange(n):
    op, contact = raw_input().strip().split(' ')


rr = raw_input
N = int(rr())

f = sys.stdin
trie = Trie()
n = int(f.readline().strip())
for i in xrange(n):
    op, contact = f.readline().strip().split(' ')

data = sorted(data, cmp=Player.comparator)

=======================================      ====================================

TAKING INPUTS

x = int(input())
x = int(input("Enter a number: "))
name = input("What's your name? ")

my_decimalnumber = float(input("enter the number"))

arr = map(int, raw_input().split())
num1, num2 = map(int, raw_input().split())

>>> data = int(input("Enter a number: "), 8)
>>> data = int(input("Enter a number: "), 2)
data = int(input("Enter a number: "), 16)

age = raw_input("Your age? ")
age = int(raw_input("Your age? "))
programming_language = eval(raw_input("Your favourite programming languages? ")) #can enter list now,auto detect


a = [True] * limit   
a[0] = a[1] = False

sqrtn = int(round(n**0.5))
   for i in range(2, sqrtn + 1): # use `xrange()` in Python 2



print(sorted(numbers))
print('{0} {1}'.format(dict_names[key], dict_values[key])


# making 1 based index array
A = list(map(int, raw_input().strip().split(' ')))
A.insert(0, 0)

>>> a = 5
>>> li = [1, 2, 3]
>>> [a] + li  # Don't use 'list' as variable name.
[5, 1, 2, 3]



[In] range(1,10)
[Out] [1, 2, 3, 4, 5, 6, 7, 8, 9]
[In] xrange(1,10)
[Out] xrange(1,10)

for i in range(1,10):
        print i
for i in xrange(1,10):
         print i

(-x) in python is actually (size - x)
for i in xrange(size):
    s1 += matrix[i][i]
    s2 += matrix[-i-1][i]

for i in xrange(100, 120, 2): # example of even numbers between 100 and 120:

np1 = n + 1
s = list(range(np1))
sqrtn = int(round(n**0.5))



print "Script 2's name: {}".format(__name__)


future
from __future__ import absolute_import, division, print_function
from __future__ import division



mark = [True] * (end-beg+1)   # [True] in bracket is imp to make alist of true false

a = [True] * limit                          # Initialize the primality list
    a[0] = a[1] = False
    for (i, isprime) in enumerate(a):   i.e (0,false), (1,false), (2,True), (3,True), (4,True) ... enumerate does this


return yield instead of an array

print ' '.join(map(str,answer))



MAX and MIN 
nmax = -sys.maxint
nmin = +sys.maxint

INT_MAX = 4294967296
INT_MIN = -4294967296



ascii value find:
>>> ord('a')
97
>>> chr(97)
'a'
>>> chr(ord('a') + 3)
'd'
>>>


n = int(raw_input())
data = []
for i in range(n):
    name, score = raw_input().split()
    score = int(score)
    player = Player(name, score)
    data.append(player)
    
data = sorted(data, cmp=Player.comparator)
for i in data:
    print i.name, i.score

$$$$$$$$$$$ explanation $$$$$$$$$$$$$$$$

>>> data = int(input("Enter a number: "), 8)
Enter a number: 777
>>> data
511

data = int(input("Enter a number: "), 16)
Enter a number: FFFF
>>> data
65535

The second parameter tells what is the base of the numbers entered and then internally it understands and converts it. If the entered data is wrong it will throw a ValueError.

>>> data = int(input("Enter a number: "), 2)
Enter a number: 1234
Traceback (most recent call last):
  File "<input>", line 1, in <module>
ValueError: invalid literal for int() with base 2: '1234'


In python 2.x raw_input() and input() functions always return string so you must convert them to int too.

x = int(raw_input("Enter a number: "))
y = int(input("Enter a number: "))

For multiple integer in a single line, map might be better.

arr = map(int, raw_input().split())
If the number is already known, (like 2 integers), you can use

num1, num2 = map(int, raw_input().split())

sorted(arr,reverse=True)


******

raw_input does not interpret the input. It always returns the input of the user without changes, i.e. raw. This raw input can be changed into the data type needed for the algorithm. To accomplish this we can use either a casting function or the eval function.

age = raw_input("Your age? ")
print(age, type(age))
('38', <type 'str'>)

age = int(raw_input("Your age? "))
Your age? 42
>>> print(age, type(age))
(42, <type 'int'>)

programming_language = eval(raw_input("Your favourite programming languages? "))
Your favourite programming languages?  ["Python", "Lisp","C++"]
>>> print(programming_language, type(programming_language))
(['Python', 'Lisp', 'C++'], <type 'list'>)

>>> programming_language = raw_input("Your favourite programming languages? ")
Your favourite programming languages? ["Python", "Lisp","C++"]
>>> print(programming_language, type(programming_language))
('["Python", "Lisp","C++"]', <type 'str'>)

programming_language = list(raw_input("Your favourite programming languages? "))
Your favourite programming languages?  ["Python", "Lisp","C++"]
>>> print(programming_language, type(programming_language))
([' ', '[', '"', 'P', 'y', 't', 'h', 'o', 'n', '"', ',', ' ', '"', 'L', 'i', 's', 'p', '"', ',', '"', 'C', '+', '+', '"', ']'], <type 'list'>)
>>> 


In python 3.x, use input() instead of raw_input()
Python 3.x has input() function which returns always string.So you must convert to int


for i in xrange(100, 120, 2): # example of even numbers between 100 and 120:
The reason why xrange was removed was because it is basically always better to use it, and the performance effects are negligible. So Python 3.x's range function is xrange from Python 2.x.



*************************************************    ***************************************  *************************



OUTPUTS

print(x)

name = input('Enter Your Name: ')
print('Hello ', name)

print str(errormessage)

print "O/p is -:",d


print(sorted(numbers))

print "Script 2's name: {}".format(__name__)
print('{0} {1}'.format(dict_names[key], dict_values[key])

__repr__ should return a printable representation of the object, most likely one of the ways possible to create this object. See official documentation here. __repr__ is more for developers while __str__ is for end users.

>>> class Point:
...   def __init__(self, x, y):
...     self.x, self.y = x, y
...   def __repr__(self):
...     return 'Point(x=%s, y=%s)' % (self.x, self.y)
>>> p = Point(1, 2)
>>> p
Point(x=1, y=2)

print format(t, '.20f')


printing in same line

import sys
sys.stdout.write(str(root.data)+ ' ')
3 5 1 4 2 6

def preOrder(root):
    global list
    if root == None:
        return
    #print root.data
    sys.stdout.write(str(root.data)+ ' ')
    #list.append(root.data)
    preOrder(root.left)
    preOrder(root.right)



class Word(object):
    def __init__(self, string, index):
        self.string = string
        self.index = index
 
# Create a DupArray object that contains an array
# of Words
def createDupArray(string, size):
    dupArray = []
 
    # One by one copy words from the given wordArray
    # to dupArray
    for i in xrange(size):
        dupArray.append(Word(string[i], i))
 
    return dupArray
# Given a list of words in wordArr[]
def printAnagramsTogether(wordArr, size):
    # Step 1: Create a copy of all words present in
    # given wordArr.
    # The copy will also have orignal indexes of words
    dupArray = createDupArray(wordArr, size)
 
    # Step 2: Iterate through all words in dupArray and sort
    # individual words.
    for i in xrange(size):
        dupArray[i].string = ''.join(sorted(dupArray[i].string))
 
    # Step 3: Now sort the array of words in dupArray
    dupArray = sorted(dupArray, key=lambda k: k.string)
 
    # Step 4: Now all words in dupArray are together, but
    # these words are changed. Use the index member of word
    # struct to get the corresponding original word
    for word in dupArray:
        print wordArr[word.index],

*************************************************    ***************************************  *************************

IF ELSE LOOPS array 

while x!= 42:

if input("Play again? ") == "no":
        break


for i in range(len(num)):

num1 = int(num[:i])
num2 = int(num[i+1:])


for (i, isprime) in enumerate(a):
        if isprime:
            yield i
            for n in xrange(i*i, limit, i):     # Mark factors non-prime
                a[n] = False


a = [True] * limit   


C.append(B[j])
C.extend(A[i:])
M.append(A.pop(0)) 




sqrtn = int(round(n**0.5))
    for i in range(2, sqrtn + 1): # use `xrange()` in Python 2



range(x,y) returns a list of each number in between x and y if you use a for loop, then range is slower. In fact, range has a bigger Index range. range(x.y) will print out a list of all the numbers in between x and y

xrange(x,y) returns xrange(x,y) but if you used a for loop, then xrange is faster. xrange has a smaller Index range. xrange will not only print out xrange(x,y) but it will still keep all the numbers that are in it.

[In] range(1,10)
[Out] [1, 2, 3, 4, 5, 6, 7, 8, 9]
[In] xrange(1,10)
[Out] xrange(1,10)
If you use a for loop, then it would work

[In] for i in range(1,10):
        print i
[Out] 1
      2
      3
      4
      5
      6
      7
      8
      9
[In] for i in xrange(1,10):
         print i
[Out] 1
      2
      3
      4
      5
      6
      7
      8
      9
There isn't much difference when using loops, though there is a difference when just printing it!




####################################################################################################

from bisect import bisect_left

def binsearch(l,e):
    '''
    Looks up element e in a sorted list l and returns False if not found.
    '''
    index = bisect_left(l,e)
    if index ==len(l) or l[index] != e:
        return False
    return index



















*************************************************    ***************************************  *************************





*************************************************    ***************************************  *************************

COLLECTIONS

try:
    input = raw_input
except NameError:
    pass
except Exception ,e:
    print "Program halted incorrect data entered",type(e)
    raise StopIteration

def prompt(message, errormessage, isvalid):
	"""Prompt for input given a message and return that value after verifying the input.

    Keyword arguments:
    message -- the message to display when asking the user for the value
    errormessage -- the message to display when the value fails validation
    isvalid -- a function that returns True if the value given by the user is valid
    """
    res = None
    while res is None:
        res = input(str(message)+': ')

sqrtn = int(round(n**0.5))
    for i in range(2, sqrtn + 1): # use `xrange()` in Python 2


raise Exception('no solution')





*************************************************    ***************************************  *************************



INBUILD FUNCTIONS


print(age, type(age))

a = [True] * limit   

import sys
>>> sys.version


r = raw_input("Enter number:")
        if r.isdigit():
	        d = int(r)/i
	            print "O/p is -:",d


p=raw_input()
    p=p.split()      
    for i in p:
        a.append(int(i))


for (i, isprime) in enumerate(a):
        if isprime:
            yield i
            for n in xrange(i*i, limit, i):     # Mark factors non-prime
                a[n] = False


import math
t=open(sys.argv[1],'r').readlines()

print(sorted(numbers))


raise StopIteration


data = sorted(data, cmp=Player.comparator) #comparator below this

class Player:
    def __init__(self, name, score):
        self.name = name
        self.score = score
        
    def __repr__(self):
        pass
        
    def comparator(a, b):
        if a.score > b.score:
            return -1
        elif a.score < b.score:
            return +1
        else:
            if a.name > b.name:
                return +1
            else:
                return -1

****************************************************************************************************************************


IMPORTANT POINTS

laways return yield insted of an arry .. It's falster and use less memory

def sieve(N):   # N will be sqrt(Real_N) i.e N = 10, then sqrt(N) will be 3
    yield 2
    D, q = {}, 3
    while q <= N:
        p = D.pop(q, 0)
        if p:
            x = q + p
            while x in D: x += p
            D[x] = p
        else:
            yield q
            D[q*q] = 2*q
        q += 2
    raise StopIteration

def primes_sieve2(limit):   # this will get accepted
    a = [True] * limit                          # Initialize the primality list
    a[0] = a[1] = False

    for (i, isprime) in enumerate(a):
        if isprime:
            yield i
            for n in xrange(i*i, limit, i):     # Mark factors non-prime
                a[n] = False


===================
Use a mutable DS to simulate pass by reference.

count = [0]
print merge_sort(unsorted, count)
print count[0]

****

count = 0

def merge_sort(li):

    if len(li) < 2: return li 
    m = len(li) / 2 
    return merge(merge_sort(li[:m]), merge_sort(li[m:])) 

def merge(l, r):
    global count    # reference global count bad for thread safety
    result = [] 
    i = j = 0 
    while i < len(l) and j < len(r): 
        if l[i] < r[j]: 
            result.append(l[i])
            i += 1 
        else: 
            result.append(r[j])
            count = count + (len(l) - i)
            j += 1
    result.extend(l[i:]) 
    result.extend(r[j:]) 
    return result

unsorted = [10,2,3,22,33,7,4,1,2]
print merge_sort(unsorted)
print count

#################

Tutorials in Python

Full graph tutorialin python
http://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
https://github.com/codingjester/code-samples/blob/master/algos/python/dfs.py
https://stackoverflow.com/questions/19472530/representing-graphs-data-structure-in-python


http://www.python-course.eu/graphs_python.php
http://www.bogotobogo.com/python/python_graph_data_structures.php


https://github.com/algorhythms/HackerRankAlgorithms



Graph:
Graph class rep : matrix and adjency list (for directed and undericted graph)
graph class min operation func (matrix and adjency list) : add, remove, indegree, outdegree, isConnected

Graph Algos:
DFS (matrix, list)
BFS (matrix, list)
shortest path Algo's

Prims
Kruskal
Dijsktra
flowd vorshel: all pair shortest path
connected components


Software Engineer (Web) at Altitude Labs https://angel.co/wavecell/jobs/50652-engineering-full-stack-engineer-singapore?src=rec
Engineering – Full Stack Engineer - Singapore at Wavecell https://angel.co/altitude-labs/jobs/192580-software-engineer-web?src=rec
https://angel.co/giveasia/jobs/226046-full-stack-developer












