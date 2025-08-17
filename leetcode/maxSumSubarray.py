def maxSumSubarray(arr, k):
    if len(arr) < k:
        return "None"
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(len(arr)-k):
        window_sum = window_sum - arr[i] + arr[i+k]
        max_sum = max(max_sum, window_sum)
        
    return max_sum

arr = [-30, 10, 50, 201, -500, -20, 4, 10, -50, 23, -44]
k = 4
print(maxSumSubarray(arr, k))