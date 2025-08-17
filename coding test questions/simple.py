#sum
a, b = 1, 2
print(a + b)  # Output: 3
#even odd
def even(n):
    return n % 2 == 0
#reverse string
def reverse_string(s):
    return s[::-1]

def check_palindrome(s):
    return s == s[::-1]

def second_largest(nums):
    unique = list(set(nums))
    if len(unique) < 2:
        return None
    unique.sort()
    return unique[-2]

def check_if_anagrams(s1, s2):
    return sorted(s1) == sorted(s2)

def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]

def find_pairs(group, target):
    pairs = set()
    seen = set()
    for num in group:
        complement = target - num
        if complement in seen:
            pairs.add(tuple(sorted(num, complement)))
        seen.add(num)
    return list(pairs)
    
def longest_substring(s):
    index = {}
    start, max_len = 0, 0
    for i, c in enumerate(s):
        if c in index and index[c] >= start:
            start = index[c] + 1
        index[c] = i
        max_len = max(max_len, i - start + 1)
    return max_len
#transpose 2d list
def transpose(matrix):
    return [list(row) for row in zip(*matrix)]