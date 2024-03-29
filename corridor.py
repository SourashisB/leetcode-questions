class Solution(object):
    def __init__(self, sampleString):
        self.sampleString = sampleString

    def numberOfWays(self, corridor):
        """
        :type corridor: str
        :rtype: int
        """
        numSeats = 0
        numPlants = 0
        dividers = 1
        mod = 1000000007
        
        for i in corridor:
            if i == 'S':
                numSeats += 1
            if numSeats == 2 and i == 'P':
                numPlants += 1
            if numSeats == 3:
                dividers *= (numPlants + 1)
                dividers %= mod
                numSeats = 1
                numPlants = 0
        
        if numSeats < 2: return 0
        return dividers
    
testsolution = Solution("SPSPPSSPSSSS")

print(testsolution.numberOfWays(testsolution.sampleString))