from typing import List

def maxSubarraySum(intArray: List[int]):
    positives = [num for num in intArray if num>= 0]
    negatives = [num for num in intArray if num<= 0]
    if len(positives) == 0:
        return max(negatives)
    else:
        return sum(positives)
    
sampleArray = [-5,-2,-5,-8,-10,-50,-8,-9]
sampleArray2 = [3,5,10,15,8,10]
sampleArray3 = [3,5,10,15,8,10,-5,-2,-5,-8,-10,-50,-8,-9]
print(maxSubarraySum(sampleArray2))
