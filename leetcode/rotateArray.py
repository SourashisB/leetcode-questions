from typing import List
def rotate( nums: List[int], k: int) -> None:
    for i in range(k):
        popped = nums.pop()
        nums.insert(0, popped)
    return nums

arr = [5,1,3,50,23,2]

print(rotate(arr,3))