from typing import List
def minSubArrayLen(target: int, nums: List[int]) -> int:
    if target in nums:
        return 1
    if target > sum(nums):
        return 0
    current_sum = 0
    start = 0
    end = 0
    min_length = -1
    for end in range(len(nums)):
        current_sum += nums[end]
        
        while current_sum >= target:
            min_length = min(min_length, end - start+1)
            current_sum -= nums[start]
            start += 1
            
    return min_length if min_length != -1 else 0
    
    
    