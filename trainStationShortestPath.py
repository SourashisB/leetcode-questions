from collections import deque
class Station:
    def __init__(self, name):
        self.name = name
        self.neighbours = []

    def add_connection(self, station):
        self.neighbours.append(station)
        station.neighbours.append(self)

class TrainMap:
    def __init__(self, name) -> None:
        self.name = name
            
    def findShortestRoute(start_station, end_station, self):
        queue = deque([[start_station]])
        visited = set()
        
        while queue:
            
            path = queue.popleft()
            station = path[-1]
            
            if station == end_station:
                return [s.name for s in path]
            
            visited.add(station)
            
            for neighbour in station.neighbours:
                if neighbour not in visited:
                    new_path = path[:]
                    new_path.append(neighbour)
                    queue.append(new_path)
        
        return None