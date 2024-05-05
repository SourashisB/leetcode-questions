from typing import List
def productExceptSelf( nums: List[int]) -> List[int]:
    products = []
    def arrayProducts(arr):
        result = 1
        for x in arr:
            result *= x
        return result
    for numbers in nums:
        products.append(arrayProducts(nums)/nums[numbers])
    
    return products