def reverseVowels(s: str) -> str:
    word = list(s)
    vowels = "AEIOUaeiou"
    start = 0
    end = len(s)-1
    while start < end:
        while start < end and vowels.find(word[start]) == -1:
            start += 1
        while start < end and vowels.find(word[end]) == -1:
            end -= 1
        
        word[start], word[end] = word[end], word[start]
        
        start += 1
        end -= 1
    
    return "".join(word)

sample = "leetcode"

print(reverseVowels(sample))