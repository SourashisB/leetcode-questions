class Solution(object):

    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        l1 = l1[::-1]
        l2 = l2[::-1]
        i=0
        num1 = 0
        num2 = 0
        while i < len(l1):
            num1 = num1 + ( 10**(len(l1)-1-i) * l1[i])
            num2 = num2 + (10**(len(l2)-1-i)*l2[i])
            i+=1
        print (num1)
        print(num2)
        answer =[int(digit) for digit in str(num1+num2)] 
        return (answer[::-1])
        
class ListNode(object):
     def __init__(self, val=0, next=None):
         self.val = val
         self.next = next

l1 = ListNode()
print(Solution.addTwoNumbers())