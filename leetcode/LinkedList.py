class LLNode:
    def __init__(self, value, nextNode=None) -> None:
        self.value = 0
        self.nextNode = None

class LinkedList:
    def __init__(self) -> None:
        self.head = None
    
    def append(self, data):
        new_node = LLNode(data)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.nextNode:
            current = current.nextNode
        current.nextNode = new_node