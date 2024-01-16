def removeDuplicates(self, nums) -> int:
    k = len(nums)
    j = 0
    while j < (k-1):
        if nums[j] == nums[j+1] and nums[j+1] == nums[j+2]:
            k = k-1
            i = nums.pop(j+2)
            print(nums)
            nums.append(i)
            print(nums)
            j= j-1
            print(j)
        j = j +1
    return k
        

array = [0,0,1,1,1,1,2,3,3]

removeDuplicates(0, array)