class Edge(object):
    def __init__(self,dim,had=0,simple=0):
        self.dim = dim
        self.had = had % self.dim
        self.simple = simple % self.dim
    
    def numH(self) -> int:
        return self._numH

    def numR(self) -> int:
        return self._numNotH

    def isEdgePresent(self) -> bool:
        return False if (self._numH == 0 and self._numNotH == 0) else True

    def is_had_edge(self) -> bool:
        return True if (self.isEdgePresent() and self._numNotH == 0) else False
        
    def is_simple_edge(self) -> bool:
        return True if (self.isEdgePresent() and self._numH == 0) else False

    def __add__(edge1, edge2):
        assert(edge1.dim == edge2.dim)
        return Edge(edge1.dim, edge1.had + edge2.had, edge1.simple + edge2.simple)

    def __bool__(self):
        return self.isEdgePresent()

    def __str__(self):
        return "Edge(h={},s={})" % (self.had, self.simple)