def binary_search(arr, target):#O (log n)
    left, right = 0, len(arr)-1
    while left <= right:
        mid = (left+right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

arr = [4,5,6,10,50,100]
target = 10
print(binary_search(arr, target))