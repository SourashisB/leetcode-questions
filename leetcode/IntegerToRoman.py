def intToRoman(num) -> str:
    divisions = [1000,900,500,400,100,90,50,40,10,9,5,4,1]
    letters = ["M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"]
   
    roman_letters = ""
    i = 0
    while num > 0:
        for _ in range(num // divisions[i]):
            roman_letters += letters[i]
            num -= divisions[i]
        i += 1
    
    return roman_letters
            
        



print(intToRoman(1994))