def distributeCandies(self, n, limit):
    def nCk(n, k):
        if k < 0 or k > n:
            return 0
        res = 1
        for i in range(1, k + 1):
            res = res * (n - i + 1) // i
        return res
    """
    :type n: int
    :type limit: int
    :rtype: int
    """
    k = 3
    total = 0
    # Inclusion-Exclusion Principle
    for mask in range(1 << k):
        sign = (-1) ** (bin(mask).count('1'))
        sum_limits = n - (limit + 1) * bin(mask).count('1')
        if sum_limits < 0:
            continue
        total += sign * nCk(sum_limits + k - 1, k - 1)
    return total
