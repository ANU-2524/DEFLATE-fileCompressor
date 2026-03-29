from collections import Counter 
import heapq
from huffman.tree import Node

class HuffmanEncoder :
    def build_tree(self , data) :
        freq = Counter(data) 
        heap = [Node(char , freq[data]) for char in freq] 
        heapq.heapify(heap)
        
        while len(heap) > 1 : 
            left = heapq.heappop(heap) 
            right = heapq.heappop(heap)
            merged = Node(None , left.freq+right.freq) 
            merged.left = left 
            merged.right = right 
            heapq.heappush(heap , merged)
        return heap[0]
    
    def generat_code(self , root) : 
        codes = {} 
        def dfs(node , code="") :
            if node is None :
                return 
            if node.char is not None :
                codes[node.char] = code
            dfs(node.left , code+ "0")
            dfs(node.right , code+"1") 
        dfs(root) 
        return codes
    
    def encode(self , data) :
        root = self.build_tree(data)
        codes = self.generat_code(root) 
        encoded = "".join(codes[ch] for ch in data) 
        return encoded , root
     
            
            

