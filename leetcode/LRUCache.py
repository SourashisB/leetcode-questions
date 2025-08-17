from collections import OrderedDict

class LRUCache:
    
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return -1, " key doesn't exist"
        else:
            self.cache.move_to_end(key)
            return self.cache[key]
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value
        
example = LRUCache(3)

example.put(1, 4)
example.put(2, 5)
example.put(3,9)
example.put(3,10)

print(example.cache)