def threeSumClosest(nums, target):
    # Sort the array
    nums.sort()
    n = len(nums)
    closest_sum = float('inf')
    
    # Iterate through the array
    for i in range(n - 2):
        # Use two pointers to find the closest sum
        left, right = i + 1, n - 1
        
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            
            # Update the closest sum if the current one is closer
            if abs(current_sum - target) < abs(closest_sum - target):
                closest_sum = current_sum
            
            # Move pointers based on the comparison with target
            if current_sum < target:
                left += 1
            elif current_sum > target:
                right -= 1
            else:
                # If the current sum equals the target, it's the closest
                return current_sum
    
    return closest_sum

# Example usage
nums = [0, 0, 0]
target = 1
print(threeSumClosest(nums, target))  # Output: 2