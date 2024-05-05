def isPalindrome(s: str) -> bool:
    
    whitelist = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    string = ''.join(filter(whitelist.__contains__, s))
    string = string.replace(" ","")
    string = string.lower()
    s = string
    low = 0
    hi = len(s)-1
    if len(s) <= 1:
        return True    
    while low <= len(s)//2:
        if s[low] != s[hi]:
            return False
        else:
            low = low + 1
            hi = hi - 1
    return True
    

sample = "race  car"
print(isPalindrome(sample))