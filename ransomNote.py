def canConstruct(self, ransomNote: str, magazine: str) -> bool:
    ransomLetters = {}
    magLetters = {}
    for chars in ransomNote:
        if chars not in ransomLetters:
            ransomLetters[chars]= 1
        else:
            ransomLetters.update({chars: ransomLetters.get(chars)+1})
    for chars in magazine:
        if chars not in magLetters:
            magLetters[chars] = 1
        else:
            magLetters.update({chars: magLetters.get(chars)+1})
    if set(ransomLetters.keys()).issubset(set(magLetters.keys())):
        for chars in ransomLetters:
            if ransomLetters.get(chars) <= magLetters.get(chars):
                continue
            elif ransomLetters.get(chars) > magLetters.get(chars):
                return False
        return True
    else:
        return False
        
    
    
note = "aaabbccddeeff"
note2 = "11223344"
print(canConstruct(note, note2))
    
