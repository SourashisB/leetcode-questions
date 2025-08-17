from typing import List

def removeDuplicates(self, nums: List[int]) -> int:
    
    for i in range(1, len(nums)):
        if nums[i] == nums[i-1]:
            nums[i-1] = -200         
    print(nums)
    nums2 = [j for j in nums if j != -200]
    nums = nums2
    print(nums)
    return len(nums)

array = [1,1,1,2]

removeDuplicates(0, array)