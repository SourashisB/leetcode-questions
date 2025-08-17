def reverseWords(s: str) -> str:
    words = s.split()
    reversedWords = words[::-1]
    reversedSentence = ' '.join(reversedWords)
    return reversedSentence