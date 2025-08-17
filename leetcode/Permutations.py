def permuteUnique(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    def backtrack(path, used):
            if len(path) == len(nums):
                result.append(path[:])
                return
            for i in range(len(nums)):
                # Skip used elements
                if used[i]:
                    continue
                # Skip duplicates: if the current number is the same as the previous,
                # and the previous hasn't been used in this path, skip it
                if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                    continue
                used[i] = True
                path.append(nums[i])
                backtrack(path, used)
                path.pop()
                used[i] = False

    nums.sort()  # Sort to handle duplicates
    result = []
    used = [False] * len(nums)
    backtrack([], used)
    return result