class Edge(object):
    def __init__(self,dim,numH=0,numNotH=0):
        self.dim = dim
        self._numH = numH % self.dim
        self._numNotH = numNotH % self.dim
    
    def get_numH(self) -> int:
        return self._numH

    def get_numNotH(self) -> int:
        return self._numNotH

    def isEdgePresent(self) -> bool:
        return False if (self._numH == 0 and self._numNotH == 0) else True

    def isAllHEdges(self) -> bool:
        return True if (self.isEdgePresent() and self._numNotH == 0) else False
        
    def isAllNotHEdges(self) -> bool:
        return True if (self.isEdgePresent() and self._numH == 0) else False

    def __add__(edge1, edge2):
        assert(edge1.dim == edge2.dim)
        return Edge(edge1.dim, edge1.get_numH() + edge2.get_numH(), edge1.get_numNotH() + edge2.get_numNotH())