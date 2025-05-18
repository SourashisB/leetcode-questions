def longestSubstring(s: str) -> int:
    l = 0
    max = 0
    char_set = set()
    for r in range(len(s)):
        while s[r] in char_set:
            char_set.remove(s[l])
            l += 1
        char_set.add(s[r])
        max = max(max, r - l + 1)
    return max