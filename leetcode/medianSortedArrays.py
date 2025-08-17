from typing import List

def medianSortedArrays(nums1: List[int], nums2: List[int]):
    nums = nums1 + nums2
    nums.sort()
    if len(nums) % 2 == 0:
        middle = len(nums)//2
        return ((nums[middle] + nums[middle + 1])/2)
    else:
        return nums[len(nums)//2]


print(medianSortedArrays([0,0,2,3,4,5,6,15],[2,3,4,8,15,20,50]))