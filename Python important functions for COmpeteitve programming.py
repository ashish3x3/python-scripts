
from collections import deque,Counter
from bisect import insort, bisect_left
from itertools import islice

from collections import defaultdict
import pprint,json
pp = pprint.PrettyPrinter(indent=4)

import collections,json
import sys
from __future__ import division

from collections import deque,Counter
from bisect import insort, bisect_left
from itertools import islice
from __future__ import absolute_import, division, print_function
from __future__ import division
import random
from Queue import PriorityQueue
import sys

from fractions import gcd
from __future__ import absolute_import, division, print_function
from ctypes import *
from collections import Counter

from string import ascii_lowercase
from string import ascii_uppercase
from random import randint
from hashlib import md5


import math
import sys
from bisect import bisect

# importing based on version
if sys.version_info >= (2, 5):
    import hashlib
    md5_constructor = hashlib.md5
else:
    import md5
    md5_constructor = md5.new



***********************************************************************************
from __future__ import division
import math,sys
from fractions import gcd
from collections import defaultdict
mod = 10**9 + 7


T = int(raw_input())

for _ in xrange(T):
    a,b = map(int,raw_input().strip().split())





takinf string:
s = raw_input().strip()

mp = defaultdict(list)

(-x) in python is actually (size - x)
**************************************************************************************

802  inverse of modulo
808  factorial % n
816  compute a^n
808  factorial using range

940 Printng in same line without brackets
995 iterative ncr function
1087 finding subsets of a set using itertools
1128    solve_quadratic

1206  two_smallest numbers n list

1250 converting string to list
1259 combination lexiographic order using itertools
1285 binary rep of number using bin(i)
1290 printing powerset
1305 concatinating string, concatination, combining, reverse list
1356  number to word conversion

1022    factorial pre computation

1438 converting string to list
1446 ascii value find
1476 finding common prefix in 2 string
1530 common prefix in 2 string
1534 common substring exists
1549 converting matrix/grid/2D to string
1557 converting each index to its row and column from string rep
1571 searching in 2d sorted matrix using binary search
1644 global variabel access in python

1659 Deque IN PYTHON
1720 creating structure for custom condition storage..
1756 kadane and its variation to handle negative
























**************************************************************************************



class Stack:
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()

     def peek(self):
         return self.items[len(self.items)-1]

     def size(self):
         return len(self.items)




class BinarySearchTree:
    def __init__(self):
        self.root = None

    def create(self, val):
        if self.root == None:
            self.root = Node(val)
        else:
            current = self.root

            while True:
                if val < current.info:
                    if current.left:
                        current = current.left
                    else:
                        current.left = Node(val)
                        break
                elif val > current.info:
                    if current.right:
                        current = current.right
                    else:
                        current.right = Node(val)
                        break
                else:
                    break

def height(root):
    if root == None:
        return -1
    #lHeight = 1 + height(root.left)
    #rHeight = 1 + height(root.right)
    return 1 + max(height(root.left),height(root.right))

def DFS(self, i, j, visited, count):

        # Mark this cell as visited
        visited[i][j] = True

        self.cur_count+=1
        self.max = max(self.max, self.cur_count)

        # Recur for all connected neighbours
        for dir in self.dirs:
            if self.isSafe(i + dir[0], j + dir[1], visited):
                self.DFS(i + dir[0], j + dir[1], visited,self.cur_count)




def binSearch(arr, lo, hi):
    if lo > hi:
        return
    while lo < hi:
        mid = (lo+hi)//2
        if arr[mid] > arr[mid+1] and arr[mid] > arr[mid-1]:
            return mid
        if arr[mid] > arr[mid-1]:
            binSearch(arr, mid+1, hi)
        elif arr[mid] < arr[mid+1]:
            binSearch(arr, lo, mid-1)


arr = map(int, raw_input().strip().split(' '))

for idx, flavor in enumerate(flavors):


sorted([flavor_map[residual], idx])



def check_binary_search_tree_(root, min=-sys.maxint, max=+sys.maxint):

    if root is None:
        return True

    if root.data >= max:
        return False

    if root.data <= min:
        return False

    left = check_binary_search_tree_(root.left, min, root.data)
    right = check_binary_search_tree_(root.right, root.data, max)

    return left & right


        ################################################################################################################################
# TRIE

# trie
def insert_check(trie, str):
    exist =  False
    for letter in str:
        if letter in trie:
            exist = True
            if 'end' in trie[letter]:
                return ['BAD SET',str]
            trie = trie[letter]
        else:
            exist = False
            trie[letter] = {}
            trie = trie[letter]
    trie['end'] = 'end'
    if exist:
        return ['BAD SET',str]

def Trie():
    return collections.defaultdict(Trie)


def add(node, s):
    if 'count' not in node:
        node['count'] = 0
    node['count'] += 1
    # print(json.dumps(node[s[0]], indent=1))
    # print 's ',s
    # print 'len(s) ',len(s)

    if s:
        add(node[s[0]], s[1:])

def find(node, s):
    if s and node:
        return find(node[s[0]], s[1:])
    else:
        return node.get('count', 0)


contacts = Trie()
n = int(raw_input().strip())
for a0 in xrange(n):
    op, contact = raw_input().strip().split(' ')
    if op == 'add':
        add(contacts, contact)
        print(json.dumps(contacts, indent=1))
    else:
        print find(contacts, contact)

class Trie(object):

    def __init__(self):
        self.items = {}

    def insert(self, value):
        current_dict = self.items
        for letter in value:
            if letter not in current_dict:
                current_dict[letter] = {}
            if 'count' not in current_dict:
                current_dict['count'] = 1
            else:
                current_dict['count'] += 1
            current_dict = current_dict[letter]

        current_dict['end'] = 'end'
        current_dict['count'] = 1

    def partial_search(self, value):
        current_dict = self.items
        for letter in value:
            if letter in current_dict:
                current_dict = current_dict[letter]
            else:
                return 0

        return current_dict['count']

tries = [0, {}]

def add(name):
    root = tries
    for c in name:
        if c not in root[1]:
            root[1][c] = [0, {}]
        root[1][c][0] += 1
        root = root[1][c]

def find(name):
    root = tries

    for e in name:
        if e not in root[1]:
            return 0
        root = root[1][e]
    return root[0]

n = int(raw_input().strip())
for a0 in range(n):
    op, contact = raw_input().strip().split(' ')
    if op == 'add':
        add(contact)
    if op == 'find':
        print find(contact)



from collections import defaultdict as ddic

def _trie():
    return ddic(_trie)

trie = _trie()

COUNT, END = True, False

def insert(query):
    cur = trie
    for letter in query:
        cur[COUNT] = cur.get(COUNT, 0) + 1
        cur = cur[letter]
    cur[COUNT] = cur.get(COUNT, 0) + 1
    cur[END] = END

def search(prefix):
    cur = trie
    for letter in prefix:
        if letter not in cur:
            return 0
        cur = cur[letter]
    return cur[COUNT]

class Trie:
    """
    Implement a trie with insert, search, and startsWith methods.
    """
    def __init__(self):
        self.root = defaultdict()

    def printRoot(self):
        # print type(self.root)
        # print pprint.pprint(dict(self.root), indent=1)
        print(json.dumps(self.root, indent=1))
        # print sum(len(v) for v in self.root.itervalues())
        # print len(self.root)


    # @param {string} word
    # @return {void}
    # Inserts a word into the trie.
    def insert(self, word):
        # print 'word ', word
        current = self.root
        # print 'current ',pprint.pprint(current)
        for letter in word:
            # print 'letter ', letter
            current = current.setdefault(letter, {})
            # print 'current ',pprint.pprint(current)
        current.setdefault("_end")

    # @param {string} word
    # @return {boolean}
    # Returns if the word is in the trie.
    def search(self, word):
        current = self.root
        for letter in word:
            if letter not in current:
                return False
            current = current[letter]
        if "_end" in current:
            return True
        return False

    # @param {string} prefix
    # @return {boolean}
    # Returns if there is any word in the trie
    # that starts with the given prefix.
    def startsWith(self, prefix):
        if len(prefix)==0:
            return True
        current = self.root
        for letter in prefix:
            if letter not in current:
                print 0
                return False
            current = current[letter]
            # print 'new current'
            # print(json.dumps(current, indent=1))
        # print sum(len(v) for v in current.itervalues())
        print len(current)
        return True


        ################################################################################################################################

def rightViewUtil(root, level, max_level):

    # Base Case
    if root is None:
        return

    # If this is the last node of its level
    if (max_level[0] < level):
        print "%d   " %(root.data),
        max_level[0] = level

    # Recur for right subtree first, then left subtree
    rightViewUtil(root.right, level+1, max_level)
    rightViewUtil(root.left, level+1, max_level)


l = []
for _ in xrange(n):
    bisect.insort(l, input())
    print '{:.1f}'.format(medium(l))


d = deque(seq[0:M])

Counter(d).most_common(1)[0][0]





def factors(n):
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def factors(n):
    results = set()
    for i in xrange(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            results.add(i)
            results.add(int(n/i))
    return results
def factors(n):
    results = set()
    count = 0
    for i in xrange(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            print '...'
            print 'i ', i
            print 'n/i ',int(n/i)
            print '...'

            if i % 2 == 0:
                count+=1
                results.add(i)
            if int(n/i) % 2 == 0 and int(n/i) not in results:
                count+=1
                results.add(int(n/i))

def divisorGenerator(n):        # print list(divisorGenerator(100))
    large_divisors = []
    for i in xrange(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor


def sum_digits(n):
    r = 0
    while n:
        r, n = r + n % 10, n // 10
    return r

def digitsum(x):
    return sum(map(int,str(x)))



mp = defaultdict(list)





x = gcd(a, b)

def getGCD(n, m):
    if m == 0:
        return n
    else:
        return getGCD(m, n%m)


m = int(math.ceil(N/2))

# overlaping of rectnagel: ((sqrt(2)*Diagonal(big))-(sqrt(2)*Diagonal(small))/(L1-L2)
t = float((math.sqrt(2)*(par[0]-math.sqrt(Aq)))/(abs(par[1]-par[2])))

res  = getGCD(int(math.fabs(x2-x1)), int(math.fabs(y2-y1)))

#passoble path exist between 2 point in grid
if getGCD(x,y) == getGCD(a,b):



def oddEven(x,y):
    if x%2 == 0 and y%2 != 0:
        return True
    elif x%2 != 0 and y%2 == 0:
        return True

    return False

# prime factors of anumber
# The primefac module does factorizations with all the fancy techniques mathematicians have developed over the centuries:
# [factor for factor in primefac.primefac(2016)] returns a list of factors: [2, 2, 2, 2, 2, 3, 3, 7]
# list(primefac.primefac(2016))
import primefac
import sys

n = int( sys.argv[1] )
factors = list( primefac.primefac(n) )
print '\n'.join(map(str, factors))



# Most of the above solutions appear somewhat incomplete. A prime factorization would repeat each prime factor of the number (e.g. 9 = [3 3]).
# Also, the above solutions could be written as lazy functions for implementation convenience.
# The use sieve Of Eratosthenes to find primes to test is optimal, but; the above implementation used more memory than necessary.
# I'm not certain if/how "wheel factorization" would be superior to applying only prime factors, for division tests of n.
# While these solution are indeed helpful, I'd suggest the following two functions -

# Function-1 :
def primes(n):
    if n < 2: return
    yield 2
    plist = [2]
    for i in range(3,n):
        test = True
        for j in plist:
            if j>n**0.5:   # test till sqrt(N), j here loops over actual prime number and not index [2,3,5,7]
                break
            if i%j==0:     # if i=9 got divisible by any number present in the primeList[2,3,5,7] then don't add 9 to list..aff none than add 9 to list
                test = False
                break
        if test:
            plist.append(i)
            yield i

# Function-2 :
def pfactors(n):
    for p in primes(n):
        while n%p==0:   # of all prime return, check if num%prime==0,those are its prime factors
            yield p     # this will prime multiple occurence of prime no.. like 12 = 2,2,3  9 = 3,3 8 = 2,2,2
            n=n//p      # n = 12/2=6/2=>3/3 =>! retrun ; 9 = 9/3=>3/3=>1 return ; 8 = 8/2=>4/2=>2/2=>1 return
            if n==1: return

# list(pfactors(99999))
# [3, 3, 41, 271]

# 3*3*41*271
# 99999

# list(pfactors(13290059))
# [3119, 4261]


# return prime numbers and not factors
def sieve(maxNum):
    yield 2
    D, q = {}, 3
    while q <= maxNum:
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

# return prime numbers and not factors
def primes_sieve2(limit):
    a = [True] * limit                          # Initialize the primality list
    a[0] = a[1] = False

    for (i, isprime) in enumerate(a):
        if isprime:
            yield i
            for n in xrange(i*i, limit, i):     # Mark factors non-prime
                a[n] = False

def prime_eratosthenes(n):
    non_prime_list = []
    prime_list = []
    for i in range(2, n+1):
        if i not in non_prime_list:
            print (i)
            prime_list.append(i)
            for j in range(i*i, n+1, i):
                non_prime_list.append(j)

    return prime_list


# segmented seive
def prime_num_between_gen(end):

    if end == 2:
        return 1

    res=1
    count = 0
    limit = int(math.floor(math.sqrt(end))) + 1
    #print limit
    prime = sieve(int(math.sqrt(end)))

    for p in prime:
        if res*p<=end:
            count+=1
            res*=p
        else:
            return count

    low = limit
    high = 2*limit

    while low < end:
        mark = [True] * (limit+1)
        for p in prime:
            lo_lim = int(math.floor((low/p)))*p

            for j in xrange(lo_lim, high, p):
                mark[j - low] = False

        for k in xrange(low, high):
            #print k
            if mark[k - low] == True:
                if res*k<=end:
                    count+=1
                    res*=k
                else:
                    return count

        low = low+limit
        high = high+limit
        if high >= end:
            high = end

    return count


def prime(x):
    x = x + 1
    primes = []
    for a in range(1, 10000):
        for b in range(2, a):
            if a % b == 0: break
        else:
            primes.append(a)
        if len(primes) == x:
            return primes[1:]

# factors of a number
def print_factors(x):
   for i in range(1, x + 1):
       if x % i == 0:
           yield i

def factors(n):
    return set(reduce(list.__add__,([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

# roatre ponts by 180 degree
 # transform asix (x,y) --> (x-h,y-k) if (h,k) is new origin
    newPx = Px-Qx
    newPy = Py-Qy

    # roatating by 180 (x,y)-->(-x,-y)
    t_Px = -newPx
    t_Py = -newPy

    # final ansewer (transformed t_Px,t_Py) --> (t_Px + h, t_Py+k) if (h,k) is new origin



print ' '.join(map(str,answer))




def array_left_rotation(a, n, d):
    a = rev(a, 0, d)
    # print a
    a = rev(a, d, n)
    # print a
    a = rev(a, 0, n)
    # print a

    return a

def rev(a, beg, end):
    limit = (end-beg)/2
    # print 'beg,end ',beg,end
    # print 'limit ',limit
    last = end
    for i in xrange(beg, beg+limit):
        # print 'before swap ',a
        #swap a[i] with a[end-1]
        a[i], a[last-1] = a[last-1], a[i]
        last = last-1
        # print 'after swap ',a

    return a




def fibonacci(a,b,n):
    a = a
    b = b
    if n < 0:
        print("Incorrect input")
    elif n == 0:
        return a
    elif n == 1:
        return b
    else:
        for i in xrange(2,n+1):
            c = a + b
            #print i,c
            a = b
            b = c
        return b

def fib0(n):
    v1, v2, v3 = 1, 1, 0    # initialise a matrix [[1,1],[1,0]]
    for rec in bin(n)[3:]:  # perform fast exponentiation of the matrix (quickly raise it to the nth power)
        calc = v2*v2
        v1, v2, v3 = v1*v1+calc, (v1+v3)*v2, calc+v3*v3
        if rec=='1':    v1, v2, v3 = v1+v2, v1, v2
    return v2

global fibs
def fib(n):
    print fibs
    if n in fibs: return fibs[n]
    if n % 2 == 0:
        fibs[n] = ((2 * fib((n / 2) - 1)) + fib(n / 2)) * fib(n / 2)
        return fibs[n]
    else:
        fibs[n] = (fib((n - 1) / 2) ** 2) + (fib((n+1) / 2) ** 2)
        return fibs[n]



# (Public) Returns F(n).
def fibonacci_fast(n):
    if n < 0:
        raise ValueError("Negative arguments not implemented")
    return _fib(n)[0]


# (Private) Returns the tuple (F(n), F(n+1)).
def _fib(n):
    if n == 0:
        return (0, 1)
    else:
        a, b = _fib(n // 2)
        c = a * (b * 2 - a)
        d = a * a + b * b
        if n % 2 == 0:
            return (c, d)
        else:
            return (d, c + d)

mod = 1000000007
def mult(x,y):
    if ( len(y) == 2 ):
        a = ((x[0]*y[0])%mod+(x[1]*y[1])%mod)%mod
        b = ((x[2]*y[0])%mod+(x[3]*y[1])%mod)%mod
        return [a,b]
    a = ((x[0]*y[0])%mod + (x[1]*y[2])%mod)%mod
    b = ((x[0]*y[1])%mod+ (x[1]*y[3])%mod)%mod
    c = ((x[2]*y[0])%mod + (x[3]*y[2])%mod)%mod
    d = ((x[2]*y[1])%mod + (x[3]*y[3])%mod)%mod
    return [a,b,c,d]

# Modular exponentiation with iterative binary exponentiation
def expmod_iter(a,b,c):
    x = 1
    while(b>0):
        if(b&1==1):
            x = (x*a)%c
        a=(a*a)%c
        b >>= 1
    return x%c

# Only works for positive powers!
def matrix_power( x, n ):
    if ( n == 1 or n == 0 ):
        return x
    if ( n%2 == 0 ):
        return matrix_power( mult(x, x), n//2 )
    return mult(x, matrix_power( mult(x, x), n//2 ) )


print mult(matrix_power(A,n-1),v)[0]

def binary_exponent(base, exponent):
    """\
    Binary Exponentiation

    Instead of computing the exponentiation in the traditional way,
    convert the exponent to its reverse binary representation.

    Each time a 1-bit is encountered, we multiply the running total by
    the base, then square the base.
    """
    # Convert n to base 2, then reverse it. bin(6)=0b110, from second index, reverse
    exponent = bin(exponent)[2:][::-1]

    result = 1
    for i in xrange(len(exponent)):
        if exponent[i] is '1':
            result *= base
        base *= base
    return result


# ******
from itertools import compress
from operator import mul

def radix(b) :
    while b :
        yield b & 1
        b >>= 1

def squares(b) :
    while True :
        yield b
        b *= b

def fast_exp(b, exp) :
    return reduce(mul, compress(squares(b), radix(exp)), 1)

# ******

def _digits_of_n(n, b):
    """ Return the list of the digits in the base 'b'
        representation of n, from LSB to MSB
    """
    digits = []

    while n:
        digits.append(n % b)
        n /= b

    return digits

# base 2 bits
def _bits_of_n(n):
    """ Return the list of the bits in the binary
        representation of n, from LSB to MSB
    """
    bits = []

    while n:
        bits.append(n % 2)
        n /= 2

    return bits

#Modular exponentiation by squaring
#Here's the right-to-left method with modular reductions at each step.

def modexp_rl(a, b, n):
    r = 1
    while 1:
        if b % 2 == 1:
            r = r * a % n
        b /= 2
        if b == 0:
            break
        a = a * a % n

    return r
#We use exactly the same algorithm, but reduce every multiplication . So the numbers we deal with here are never very large.

#Similarly, here's the left-to-right method:

def modexp_lr(a, b, n):
    r = 1
    for bit in reversed(_bits_of_n(b)):
        r = r * r % n
        if bit == 1:
            r = r * a % n
    return r

# http://eli.thegreenplace.net/2009/03/28/efficient-modular-exponentiation-algorithms
def modexp_lr_k_ary(a, b, n, k=5):
    """ Compute a ** b (mod n) in base K system

        K-ary LR method, with a customizable 'k'.
    """
    base = 2 << (k - 1)

    # Precompute the table of exponents
    table = [1] * base
    for i in xrange(1, base):
        table[i] = table[i - 1] * a % n

    # Just like the binary LR method, just with a
    # different base
    #
    r = 1
    for digit in reversed(_digits_of_n(b, base)):
        for i in xrange(k):
            r = r * r % n
        if digit:
            r = r * table[digit] % n

    return r


For n >= 1, derive the identity
nC0 + nC1 + nC2 + ... + nCn = 2^n
[Hint: Let a = b = 1 in the binomial theorem]
nCn = 1 and nC0 = 1.
nCr = nC(n - r)


# Binary
>>> "{0:b}".format(10)
'1010'

>>> bin(10)
'0b1010'

str(bin(i))[2:]

get_bin = lambda x: format(x, 'b')

print(get_bin(3))
>>> '11'

print(get_bin(-3))
>>> '-11'


When you want a n-bit representation:

get_bin = lambda x, n: format(x, 'b').zfill(n)
>>> get_bin(12, 32)
'00000000000000000000000000001100'
>>> get_bin(-12, 32)
'-00000000000000000000000000001100'

j = int(bin(i)[2:].replace('1','9'))    # converting 10001 to 90001 % M(4,5,6,7,etc) checking a number X made of 9 and 0 which is multiple of N
        if j % c == 0:
            break
'''
The binary to decimal conversion algorithm is of complexity O(k), i.e. the complexity of the loop. However, k is a count of the number of digits in the binary representation of x, that is log2x, thus this can also be treated as a algorithm of complexity O(log x)!
The decimal to binary algorithm contains a while loop that depends on the value associated with the variable identified as x, but the range is divided in half at each iteration. Thus this is similar to the binary search algorithm and is of complexity O(log x) also!
'''




t=input()
assert 1 <= t <= 10000
assert 1 <= n <= 500
assert 1 <= b <= 1e3
assert 1 <= c <= 1e3



def calculate_combinations(n, r):
    return factorial(n) // factorial(r) // factorial(n-r)

def baseN(num,b,numerals="0123456789"):
    return ((num == 0) and numerals[0]) or (baseN(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])


if n > max([int(c) for c in str(m)]):


nm = long(str(m),n)


print ' '.join(map(str,sorted(lis)))




from math import factorial

def nCr(n, r):
    return factorial(n) // factorial(r) // factorial(n-r)


# iterative ncr formula
def nCk(n, k):
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
    return res


int nCr(int n, int r) {
    if (n < r) {
        return 0;
    }

    if (n - r < r) {
        r = n - r;
    }

    int prod = 1;
    for (int div = 1; div <= r; div++) {
        prod *= n;
        prod /= div;

        n--;
    }

    return prod;
}

# making 1 based index array
A = list(map(int, raw_input().strip().split(' ')))
A.insert(0, 0)

>>> a = 5
>>> li = [1, 2, 3]
>>> [a] + li  # Don't use 'list' as variable name.
[5, 1, 2, 3]


#MAX and MIN
nmax = -sys.maxint
nmin = +sys.maxint
INT_MAX = 4294967296
INT_MIN = -4294967296


# factorial pre computation
fa = [1]
for i in xrange(1,300):
    fa.append((i*fa[i-1]))


# inverse of modulo
'''
Inverse
Fermat's Little theorem says that
a^(p-1) == 1 mod p or a^(p-1) % p == 1

where p is prime
      gcd(a, p) != 1 i.e not co-prime
Multiplying both sides by a^(-1), we get
a^(p-1) * a^(-1) == a^(-1) mod p
a^(p-2) == a^(-1)  mod p
a^(-1)  == a^(p-2) mod p
'''
modulo = 7L + 10**9
mod = 10**9 + 7
def inversemod(a):
  return getpowermod(a,mod-2)

def inverse(n):
    return pow(n,p-2,p)


# factorial % n
'''
(a*b) mod m = ((a mod m)*(b mod m)) mod m
to calculate factorial as
fact(0) = 1
fact(n) = fact(n-1)*n % (10^9+7)

'''
def getfactorialmod(b):
  val = 1
  for i in range(1,b):
    val =((val%mod)*((i+1)%mod))%mod
  return val


# compute a^n
def getpowermod(a,b):
  if b==0:
    return 1
  if b == 1:
    return a
  temp = getpowermod(a,b/2)
  if b%2==0:
    return ((temp%mod)**2)%mod
  else:
    return (a*((temp%mod)**2))%mod

# factorial using range
for i in range(1, a+b-1):
          numerator = ((numerator%mod)*(i%mod))%mod


def factorial_range(a,b):
    result = 1
    for i in range(a,b):
        result *= i
        result %= p
    return result

def product(a,b):
    return (a*b) % p

# ncr
def nCr(n,r):
    return product(factorial_range(n-r+1,n+1),inverse(factorial_range(1,r+1)))

print(nCr((m-1)+(n-1),(m-1)))

# factorial precomputation
fact = [1]
n = 1
#print 2 * 10 ** 6 # 2000000
while n < 2 * 10 ** 6:
    fact.append((n * fact[-1]) % (10 ** 9 + 7))
    n += 1


fact=[1]
for x in xrange(2000000):
    fact.append( (fact[x]*(x+1))%p )


def inverse(n):
    return pow(n, 10 ** 9 + 5, 10 ** 9 + 7)  # n^p-2  # return x to the power y, modulo z

# ncr
def choose(n,k):
    return (fact[n] * inverse(fact[k]*fact[n-k])) % (10 ** 9 + 7)

print choose(m+n-2, m-1)

# ncr
def binom(n, k, p):
  if k + k > n: return binom(n, n - k, p)
  res = 1
  for i in range(k):
    res = (res * (n - i)) % p
    res = (res * pow(i + 1, p - 2, p)) % p
  return res

print(binom(m + n - 2, m - 1, P))


# extended euclid algo
def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m

# ncr
def nCr(n,r):
    return (fact[n] * modinv(fact[r],p) * modinv(fact[n-r],p) ) %1000000007


# permutation
def nPr(n,r):
    return factorial(n)//factorial(n-r)


sorted(arr,reverse=True)

data = sorted(data, cmp=Player.comparator)

for i in reversed(arr):
    sys.stdout.write(str(root.data)+ ' ')


# Printng in same line without brackets
print ' '.join(map(str,arr))




import itertools
def findsubsets(S,m):
    return set(itertools.combinations(S, m))




def subsets(nums):
  if nums is None:
    return None
  subsets = [[]]
  next = []
  for n in nums:
    for s in subsets:
      next.append(s + [n])
    subsets += next
    next = []
  return subsets


#The idea of a simple recursive solution is that if you have all subsets of an array A already generated as S = subsets(A), and now you want to go to a bigger set B which is the same as A, but has a new element x, i.e. B = A + {x}, then every subset of B is either already in S, or if it’s not there, it must contain x, and so it will be in a form of s + {x} for every s from S.
# parameter s is a list
# returns a list of all subsets of s including [] and the s itself
def subsets(s):
    # base case
    if len(s) == 0:
        return [[]]
    # the input set is not empty, divide and conquer!
    h, t = s[0], s[1:]
    ss_excl_h = subsets(t)
    ss_incl_h = [([h] + ss) for ss in ss_excl_h]
    return ss_incl_h + ss_excl_h

print subsets([1, 2, 3])
# prints: [[1, 2, 3], [1, 2], [1, 3], [1], [2, 3], [2], [3], []]




#ax^2+bx+c=0 - quadaratic equation
def solve_quadratic(a,b,c):
    d = b**2 - 4*a*c

    if d > 0:
        sd = d**(0.5)
        return [-(b+sd)/(2*a), -(b-sd)/(2*a)]
    elif d == 0:
        return [-b/(2*a)]
    return []

squareroot = {}
for g in xrange(1, 1000000):  # 10**6
    squareroot[g*g]=g;




def two_smallest(numbers):
    m1, m2 = float('inf'), float('inf')
    for x in numbers:
        if x <= m1:
            m1, m2 = x, m1
        elif x < m2:
            m2 = x
    return m1,m2





'''
Nth Prime number:
nln(n)+n(ln(ln(n))−1)
and less than
nln(n)+nln(ln(n))
When n≥6. So if you're searching for the nth prime, I'd look in this gap.

'''

prime1200 = [2,3,5,7,11,13,17,19,23,29 ,31,37,41,43,47,53,59,61,67,71 ,73,79,83,89,97,99,211,223,227,229 ,233,239,241,251,257,263,269,271,277,281 ,283,293,307,311,313,317,331,337,347,349 ,353,359,367,373,379,383,389,397,401,409 ,419,421,431,433,439,443,449,457,461,463 ,467,479,487,491,499,503,509,521,523,541 ,547,557,563,569,571,577,587,593,599,601 ,607,613,617,619,631,641,643,647,653,659 ,661,673,677,683,691,701,709,719,727,733 ,739,743,751,757,761,769,773,787,797,809 ,811,821,823,827,829,839,853,857,859,863 ,877,881,883,887,907,911,919,929,937,941 ,947,953,967,971,977,983,991,997,1009,1013 ,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069 ,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151 ,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223 ,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291 ,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373 ,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451 ,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511 ,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583 ,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657 ,1663,1667,1669,1693,1697,1699,1709,1721,1723,1733 ,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811 ,1823,1831,1847,1861,1867,1871,1873,1877,1879,1889 ,1901,1907,1913,1931,1933,1949,1951,1973,1979,1987 ,1993,1997,1999,2003,2011,2017,2027,2029,2039,2053 ,2063,2069,2081,2083,2087,2089,2099,2111,2113,2129 ,2131,2137,2141,2143,2153,2161,2179,2203,2207,2213 ,2221,2237,2239,2243,2251,2267,2269,2273,2281,2287 ,2293,2297,2309,2311,2333,2339,2341,2347,2351,2357 ,2371,2377,2381,2383,2389,2393,2399,2411,2417,2423 ,2437,2441,2447,2459,2467,2473,2477,2503,2521,2531 ,2539,2543,2549,2551,2557,2579,2591,2593,2609,2617 ,2621,2633,2647,2657,2659,2663,2671,2677,2683,2687 ,2689,2693,2699,2707,2711,2713,2719,2729,2731,2741 ,2749,2753,2767,2777,2789,2791,2797,2801,2803,2819 ,2833,2837,2843,2851,2857,2861,2879,2887,2897,2903 ,2909,2917,2927,2939,2953,2957,2963,2969,2971,2999 ,3001,3011,3019,3023,3037,3041,3049,3061,3067,3079 ,3083,3089,3109,3119,3121,3137,3163,3167,3169,3181 ,3187,3191,3203,3209,3217,3221,3229,3251,3253,3257 ,3259,3271,3299,3301,3307,3313,3319,3323,3329,3331 ,3343,3347,3359,3361,3371,3373,3389,3391,3407,3413 ,3433,3449,3457,3461,3463,3467,3469,3491,3499,3511 ,3517,3527,3529,3533,3539,3541,3547,3557,3559,3571 ,3581,3583,3593,3607,3613,3617,3623,3631,3637,3643 ,3659,3671,3673,3677,3691,3697,3701,3709,3719,3727 ,3733,3739,3761,3767,3769,3779,3793,3797,3803,3821 ,3823,3833,3847,3851,3853,3863,3877,3881,3889,3907 ,3911,3917,3919,3923,3929,3931,3943,3947,3967,3989 ,4001,4003,4007,4013,4019,4021,4027,4049,4051,4057 ,4073,4079,4091,4093,4099,4111,4127,4129,4133,4139 ,4153,4157,4159,4177,4201,4211,4217,4219,4229,4231 ,4241,4243,4253,4259,4261,4271,4273,4283,4289,4297 ,4327,4337,4339,4349,4357,4363,4373,4391,4397,4409 ,4421,4423,4441,4447,4451,4457,4463,4481,4483,4493 ,4507,4513,4517,4519,4523,4547,4549,4561,4567,4583 ,4591,4597,4603,4621,4637,4639,4643,4649,4651,4657 ,4663,4673,4679,4691,4703,4721,4723,4729,4733,4751 ,4759,4783,4787,4789,4793,4799,4801,4813,4817,4831 ,4861,4871,4877,4889,4903,4909,4919,4931,4933,4937 ,4943,4951,4957,4967,4969,4973,4987,4993,4999,5003 ,5009,5011,5021,5023,5039,5051,5059,5077,5081,5087 ,5099,5101,5107,5113,5119,5147,5153,5167,5171,5179 ,5189,5197,5209,5227,5231,5233,5237,5261,5273,5279 ,5281,5297,5303,5309,5323,5333,5347,5351,5381,5387 ,5393,5399,5407,5413,5417,5419,5431,5437,5441,5443 ,5449,5471,5477,5479,5483,5501,5503,5507,5519,5521 ,5527,5531,5557,5563,5569,5573,5581,5591,5623,5639 ,5641,5647,5651,5653,5657,5659,5669,5683,5689,5693 ,5701,5711,5717,5737,5741,5743,5749,5779,5783,5791 ,5801,5807,5813,5821,5827,5839,5843,5849,5851,5857 ,5861,5867,5869,5879,5881,5897,5903,5923,5927,5939 ,5953,5981,5987,6007,6011,6029,6037,6043,6047,6053 ,6067,6073,6079,6089,6091,6101,6113,6121,6131,6133 ,6143,6151,6163,6173,6197,6199,6203,6211,6217,6221 ,6229,6247,6257,6263,6269,6271,6277,6287,6299,6301 ,6311,6317,6323,6329,6337,6343,6353,6359,6361,6367 ,6373,6379,6389,6397,6421,6427,6449,6451,6469,6473 ,6481,6491,6521,6529,6547,6551,6553,6563,6569,6571 ,6577,6581,6599,6607,6619,6637,6653,6659,6661,6673 ,6679,6689,6691,6701,6703,6709,6719,6733,6737,6761 ,6763,6779,6781,6791,6793,6803,6823,6827,6829,6833 ,6841,6857,6863,6869,6871,6883,6899,6907,6911,6917 ,6947,6949,6959,6961,6967,6971,6977,6983,6991,6997 ,7001,7013,7019,7027,7039,7043,7057,7069,7079,7103 ,7109,7121,7127,7129,7151,7159,7177,7187,7193,7207,7211,7213,7219,7229,7237,7243,7247,7253,7283,7297 ,7307,7309,7321,7331,7333,7349,7351,7369,7393,7411 ,7417,7433,7451,7457,7459,7477,7481,7487,7489,7499 ,7507,7517,7523,7529,7537,7541,7547,7549,7559,7561 ,7573,7577,7583,7589,7591,7603,7607,7621,7639,7643 ,7649,7669,7673,7681,7687,7691,7699,7703,7717,7723 ,7727,7741,7753,7757,7759,7789,7793,7817,7823,7829 ,7841,7853,7867,7873,7877,7879,7883,7901,7907,7919, 7927, 7933, 7937, 7949, 7951, 7963, 7993, 8009, 8011,  8017, 8039, 8053, 8059, 8069, 8081, 8087, 8089, 8093, 8101, 8111,8117, 8123, 8147, 8161, 8167, 8171, 8179, 8191, 8209, 8219, 8221, 8231, 8233, 8237, 8243, 8263, 8269, 8273, 8287, 8291, 8293, 8297, 8311, 8317, 8329, 8353, 8363, 8369, 8377, 8387, 8389, 8419, 8423, 8429, 8431, 8443, 8447, 8461, 8467, 8501, 8513, 8521, 8527, 8537,8539, 8543, 8563, 8573, 8581, 8597, 8599, 8609, 8623, 8627, 8629,8641, 8647, 8663, 8669, 8677, 8681, 8689, 8693, 8699, 8707, 8713, 8719, 8731, 8737, 8741, 8747, 8753, 8761, 8779, 8783, 8803, 8807,8819, 8821, 8831, 8837, 8839, 8849, 8861, 8863, 8867, 8887, 8893,8923, 8929, 8933, 8941, 8951, 8963, 8969, 8971, 8999, 9001, 9007, 9011, 9013, 9029, 9041, 9043, 9049, 9059, 9067, 9091, 9103, 9109, 9127, 9133, 9137, 9151, 9157, 9161, 9173, 9181, 9187, 9199, 9203,9209, 9221, 9227, 9239, 9241, 9257, 9277, 9281, 9283, 9293, 9311, 9319, 9323, 9337, 9341, 9343, 9349, 9371, 9377, 9391, 9397, 9403, 9413, 9419, 9421, 9431, 9433, 9437, 9439, 9461, 9463, 9467, 9473, 9479, 9491, 9497, 9511, 9521, 9533, 9539, 9547, 9551, 9587, 9601, 9613, 9619, 9623, 9629, 9631, 9643, 9649, 9661, 9677, 9679, 9689, 9697, 9719, 9721, 9733]

# power set somewhat
for _ in xrange(T):
    N = int(raw_input())
    s = list(raw_input().strip())
    s.sort()
    l = len(s)
    res = []
    for i in xrange(l):
        for k in xrange(l-i):
            a = ''
            for j in xrange(i,i+k+1):
                a = a + s[j]+''
            #res.append(a)
            print a

    #print res



# converting string to list
>>> list('hello')
['h', 'e', 'l', 'l', 'o']

s = list(raw_input().strip())




# combination
foo = []
for i in range(1,N+1):
        #Generate combinations of length i
        combs = list(it.combinations(string,i))
#Add combinations to list
[foo.append(x) for x in combs]

#Join elements in list so they can be sorted
#This is because itertools gives back tuples
bar = [''.join(x) for x in foo]
'''
returns tuple
[('x',), ('y',), ('z',)] len 1
[('x', 'y'), ('x', 'z'), ('y', 'z')] len 2

''.join(x) operation
('x',)
==>x
('x', 'y')
==>xy
'''

def combination_(S):
    combs = set()
    for i in xrange(1, len(S) + 1):
        for comb in itertools.combinations(S, i):
            combs.add(''.join(comb))
    print '\n'.join(sorted(combs))


# printing powerset
from itertools import chain, combinations

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


# binary representation of number

for i in range(1,2**n):
    # str(bin(i))[2:] returns binary string truncation 0b from binary rep of number i
    # [::-1] reverses the returned binary number  01 becaome 10, 001 becomes 100
    test=str(bin(i))[2:][::-1]
    out=""
    for i in range(0,len(test)):
        #print 'test[i] ',test[i],s[i]
        if test[i]=='1':
            out=out+str(s[i])
    a.append(out)


# concatinating string, concatination, combining, reverse list
for i in xrange(l):
        if S[i] != S[l - 1 - i]:
            x = S[:i] + S[i + 1:]
            # x[::-1] reverses x
            if x == x[::-1]:
                a = i
            else:
                a = l - 1 - i
            break




# difference of two string
temp = Counter(s[0:len(s)/2]) - Counter(s[len(s)/2:])
    print temp
    print sum(temp.values())
    for v in temp.values():
        pass

Counter(s).values()
print "YES" if len(filter(lambda x: x & 1, Counter(s).values())) <= 1 else "NO"


'''
xaxbbbxx
xaxb   bbxx
Counter({'a': 3})
3
Counter({'a': 1})
1
-1
Counter({'m': 1, 'n': 1})
2
'''





# number to word conversion
def int_to_en(num):
    d = { 0 : 'zero', 1 : 'one', 2 : 'two', 3 : 'three', 4 : 'four', 5 : 'five',
          6 : 'six', 7 : 'seven', 8 : 'eight', 9 : 'nine', 10 : 'ten',
          11 : 'eleven', 12 : 'twelve', 13 : 'thirteen', 14 : 'fourteen',
          15 : 'fifteen', 16 : 'sixteen', 17 : 'seventeen', 18 : 'eighteen',
          19 : 'nineteen', 20 : 'twenty',
          30 : 'thirty', 40 : 'forty', 50 : 'fifty', 60 : 'sixty',
          70 : 'seventy', 80 : 'eighty', 90 : 'ninety' }
    k = 1000
    m = k * 1000
    b = m * 1000
    t = b * 1000

    assert(0 <= num)

    if (num < 20):
        return d[num]

    if (num < 100):
        if num % 10 == 0: return d[num]
        else: return d[num // 10 * 10] + ' ' + d[num % 10]

    if (num < k):
        if num % 100 == 0: return d[num // 100] + ' hundred'
        else: return d[num // 100] + ' hundred and ' + int_to_en(num % 100)

    if (num < m):
        if num % k == 0: return int_to_en(num // k) + ' thousand'
        else: return int_to_en(num // k) + ' thousand, ' + int_to_en(num % k)

    if (num < b):
        if (num % m) == 0: return int_to_en(num // m) + ' million'
        else: return int_to_en(num // m) + ' million, ' + int_to_en(num % m)

    if (num < t):
        if (num % b) == 0: return int_to_en(num // b) + ' billion'
        else: return int_to_en(num // b) + ' billion, ' + int_to_en(num % b)

    if (num % t == 0): return int_to_en(num // t) + ' trillion'
    else: return int_to_en(num // t) + ' trillion, ' + int_to_en(num % t)

    raise AssertionError('num is too large: %s' % str(num))



# iterating over map
for key, value in mp.iteritems():



# (-x) in python is actually (size - x)
for i in xrange(size):
    s1 += matrix[i][i]
    s2 += matrix[-i-1][i]




c1 = len(filter(lambda x:x>0,values_aray))




#converting string to list
>>> list('hello')
['h', 'e', 'l', 'l', 'o']



from string import ascii_lowercase
from string import ascii_uppercase
# ascii value find:
>>> ord('a')
97
>>> chr(97)
'a'
>>> chr(ord('a') + 3)
'd'
>>>

small_alph = 'abcdefghijklmnopqrstuvwxyz'
capital_alph = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

for letter in message:
    if letter.lower() in ascii_lowercase:
        if letter.islower():
            pos = ascii_lowercase.index(letter)
            encoded_message += ascii_lowercase[(pos + shift) % 26]
        else:
            pos = ascii_uppercase.index(letter)
            encoded_message += ascii_uppercase[(pos + shift) % 26]






# finding common prefix
def common_start(sa, sb):
    """ returns the longest common substring from the beginning of sa and sb """
    def _iter():
        for a, b in zip(sa, sb):
            if a == b:
                yield a
            else:
                return

    return ''.join(_iter())
>>> common_start("apple pie available", "apple pies")
'apple pie'

Or a slightly stranger way:

def stop_iter():
    """An easy way to break out of a generator"""
    raise StopIteration

def common_start(sa, sb):
    ''.join(a if a == b else stop_iter() for a, b in zip(sa, sb))


#  O(N^2).
'''
how this work is it check from i=0 with j=0..n with s1[i+j] which will check both string from every location till end..like s1[0] with s2[0], then s1[0+1](here +1 will come from j loop) with s2[1]...similarly in 2nd iteration it check both string from position 1 since the i loop will be 1..i.e s1[1+0] with s2[1],then s1[1+0] with s2[1] and so on.

this works for the longest prefix and breaks on suffixes. E.g. x = "cov_basic_as_cov_x_gt_y_rna_genes_w1000000" y = "cov_rna15pcs_as_cov_x_gt_y_rna_genes_w1000000"
'''
def longestSubstringFinder(string1, string2):
    answer = ""
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        match = ""
        for j in range(len2):
            if (i + j < len1 and string1[i + j] == string2[j]):
                match += string2[j]
            else:
                if (len(match) > len(answer)):
                    answer = match
                match = ""
    return answer

print longestSubstringFinder("apple pie available", "apple pies")
print longestSubstringFinder("apples", "appleses")
print longestSubstringFinder("bapples", "cappleses")

# Output
# apple pie
# apples
# apples


# common prefix
os.path.commonprefix([s1,s2])


# common substring exists
a=set(raw_input().strip())
b=set(raw_input().strip())
if len(a.intersection(b))>0:
    print "YES"
else:
    print "NO"


print "YES" if set(list(s1)) & set(list(s2)) else "NO"

if set.intersection(m1,m2):
        return "YES"
    return "NO"

# converting matrix/grid/2D to string
for __ in range(R):
        big.append(raw_input())
big_str = "".join(big)

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23

# converting each index to its row and column from string rep
for i in t:
    row = i / C
    col = i % C
    print row,col

t = [0, 4, 8, 10, 12, 20]
0 0
0 4
1 2
1 4



# searching in 2d sorted matrix using binary search

int m = matrix.length;
int n = matrix[0].length;

int start = 0;
int end = m*n-1;

while(start<=end){
    int mid=(start+end)/2;
    int midX=mid/n;
    int midY=mid%n;

    if(matrix[midX][midY]==target)
        return true;

    if(matrix[midX][midY]<target){
        start=mid+1;
    }else{
        end=mid-1;
    }
}




# global variable to store the maximum
global maximum

def _lis(arr , n ):

    # to allow the access of global variable
    global maximum

    # Base Case
    if n == 1 :
        return 1




Deque in python
https://docs.python.org/2/library/collections.html#collections.deque
'''
 >>> from collections import deque
>>> d = deque('ghi')                 # make a new deque with three items
>>> for elem in d:                   # iterate over the deque's elements
...     print elem.upper()
G
H
I

>>> d.append('j')                    # add a new entry to the right side
>>> d.appendleft('f')                # add a new entry to the left side
>>> d                                # show the representation of the deque
deque(['f', 'g', 'h', 'i', 'j'])

>>> d.pop()                          # return and remove the rightmost item
'j'
>>> d.popleft()                      # return and remove the leftmost item
'f'
>>> list(d)                          # list the contents of the deque
['g', 'h', 'i']
>>> d[0]                             # peek at leftmost item
'g'
>>> d[-1]                            # peek at rightmost item
'i'

>>> list(reversed(d))                # list the contents of a deque in reverse
['i', 'h', 'g']
>>> 'h' in d                         # search the deque
True
>>> d.extend('jkl')                  # add multiple elements at once
>>> d
deque(['g', 'h', 'i', 'j', 'k', 'l'])
>>> d.rotate(1)                      # right rotation
>>> d
deque(['l', 'g', 'h', 'i', 'j', 'k'])
>>> d.rotate(-1)                     # left rotation
>>> d
deque(['g', 'h', 'i', 'j', 'k', 'l'])
'''




# Function to print permutations of string with duplicates allowed
# This function takes three parameters:
# 1. String
# 2. Starting index of the string
# 3. Ending index of the string.
def permute(a, l, r):
    if l==r:
        print toString(a)
    else:
        for i in xrange(l,r+1):
            a[l], a[i] = a[i], a[l]
            permute(a, l+1, r)
            a[l], a[i] = a[i], a[l] # backtrack



creating structure for custom condition storage..
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
        print wordArr[word.index]



# Python program to find maximum contiguous subarray

The implementation handles the case when all numbers in array are negative.
Drawback of kadane is it looks for positive sum in the array..in case of all negative array,it may fail without any additional modification.. see below this algo for kadane impl..

def maxSubArraySum(a,size):

    max_so_far =a[0]
    curr_max = a[0]

    for i in range(1,size):
        curr_max = max(a[i], curr_max + a[i])
        max_so_far = max(max_so_far,curr_max)

    return max_so_far

# Driver function to check the above function
a = [-2, -3, 4, -1, -2, 1, 5, -3]
print"Maximum contiguous sum is" , maxSubArraySum(a,len(a))


# Function to find the maximum contiguous subarray
from sys import maxint
def maxSubArraySum(a,size):

    max_so_far = -maxint - 1
    max_ending_here = 0

    for i in range(0, size):
        max_ending_here = max_ending_here + a[i]
        if (max_so_far < max_ending_here):
            max_so_far = max_ending_here

        if max_ending_here < 0:
            max_ending_here = 0
    return max_so_far



# Getting index of max height from list
height.index(max(height))


# to run sum on all elem of stacks. here stacks = [deque[],deque[],deque[]]
heights = list(map(sum, stacks))


#all(iterable)
#rue - If all elements in an iterable are true
#False - If any element in an iterable is false
# all values true
l = [1, 3, 4, 5]
print(all(l))

# all values false
l = [0, False]
print(all(l))

# one false value
l = [1, 3, 4, 0]
print(all(l))



from random import randint
x_i = randint(0, len(nodes) - 1)

























































































