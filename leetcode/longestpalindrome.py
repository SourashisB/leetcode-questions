def longestPalindrome(s: str):
    length = len(s)
    
    end = 1
    for i in range(length):
        low = i - 1
        high = i
        while low >= 0 and high < length and s[low]==s[high]:
            if high - low +1 > end:
                start = low
                end = high - low + 1
            low -= 1
            high += 1
        
        low = i -1
        high = i +1
        while low >= 0 and high < length and s[low]==s[high]:
            if high - low +1 > end:
                start = low
                end = high - low + 1
            low -= 1
            high += 1
    
    print(end="")
        
longestPalindrome("forgeeksskeegfor")