def longestSubstring(s: str) -> int:
    L = 0
    max_length = 0
    char_set = set()
    
    for R in range(len(s)):
        while s[R] in char_set:
            char_set.remove(s[L])
            L += 1
        char_set.add(s[R])
        max_length = max(max_length, R - L + 1)