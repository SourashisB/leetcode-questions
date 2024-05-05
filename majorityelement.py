from typing import List
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        occurences = {}
        max_count = -1
        most_common = 0
        
        for num in nums:
            occurences[num] = occurences.get(num, 0)+1
            if occurences[num] > max_count:
                max_count = occurences[num]
                most_common = num
        return most_common