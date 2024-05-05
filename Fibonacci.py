def fibonacci_iterative(n): #O(n)
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n+1):
        a,b = b, a+b
    return b

print(fibonacci_iterative(10))

def fibonacci_memoization(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memoization(n-1, memo) + fibonacci_memoization(n-2, memo)
    return memo[n]

print(fibonacci_memoization(3))