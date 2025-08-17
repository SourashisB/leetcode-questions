def reverse_linkedList(head):
    prev = None
    current = head
    while current:
        next_node = current.next #save next node
        current.next = prev      # reverse link
        prev = current
        current = next_node
    return prev

from collections import OrderedDict
class LRU_cache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    def get(self, key):
        if key not in self.cache:
            return -1, " key doesn't exist"
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value