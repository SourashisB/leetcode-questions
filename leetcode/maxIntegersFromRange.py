def maxCount(banned, n, maxSum):
    # Convert the banned list into a set for O(1) lookup
    banned_set = set(banned)
    
    # Initialize variables to keep track of the sum and count of chosen numbers
    current_sum = 0
    count = 0
    
    # Iterate through numbers from 1 to n
    for num in range(1, n + 1):
        # Skip numbers that are banned
        if num in banned_set:
            continue
        
        # Check if adding the current number exceeds maxSum
        if current_sum + num > maxSum:
            break
        
        # Add the current number to the sum and increment the count
        current_sum += num
        count += 1
    
    return count