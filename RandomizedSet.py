
class RandomizedSet:
    
    def __init__(self):
        self.set = set()
        self.capacity = 0
        
    def insert(self, val: int) -> bool:
        if val not in self.set:
            self.set.add(val)
            self.capacity += 1
            return True
        else:
            return False
    
    def remove(self, val: int) -> bool:
        if val not in self.set:
            return False
        else:
            self.set.remove(val)
            self.capacity -= 1
            return True
    
    def getRandom(self) -> int:
        
        return self.set()