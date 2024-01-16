class Solution(object):
    def __init__(self,num):
        self.num = num

    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        answer = ""  #3612
        if num % 1000 > 1 or num % 1000 == 0:
            answer = answer+("".join("M"*(num//1000)))
        num = num - (num//1000 * 1000)
        
        print (num)

        if num == 0: return answer
        if  400: answer = answer+("".join("CD"))
        if num > num % 500 > 1 or num % 500 == 0:
            answer = answer+("".join("D"))
        num = num - 500
        if num == 0: return answer

        



        return answer
                
test = Solution(3400)

print(test.intToRoman(test.num))