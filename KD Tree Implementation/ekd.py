from __future__ import annotations
import json
import math
from typing import List

class Datum():
    def __init__(self,
                 coords : tuple[int],
                 code   : str):
        self.coords = coords
        self.code   = code
    def to_json(self) -> dict:
        dict_repr = {'coords':self.coords,'code':self.code}
        return(dict_repr)

class NodeInternal():
    def  __init__(self,
                  splitindex : int,
                  splitvalue : float,
                  leftchild,
                  rightchild):
        self.splitindex = splitindex
        self.splitvalue = splitvalue
        self.leftchild  = leftchild
        self.rightchild = rightchild

class NodeLeaf():
    def  __init__(self,
                  data : List[Datum]):
        self.data = data

class EKDtree():
    def  __init__(self,
                  splitmethod : str,
                  k           : int,
                  m           : int,
                  root        : NodeLeaf = None):
        self.k    = k
        self.m    = m
        self.splitmethod = splitmethod
        self.root = root

    # For the tree rooted at root, dump the tree to stringified JSON object and return.
    def dump(self) -> str:
        def _to_dict(node) -> dict:
            if isinstance(node,NodeLeaf):
                # Sort the data by code so you there's no ambiguity.
                sd = sorted(node.data, key = lambda x:x.code)
                return {
                    "points": [str({'coords': datum.coords,'code': datum.code}) for datum in sd]
                }
            else:
                return {
                    "splitindex" : node.splitindex,
                    "splitvalue" : node.splitvalue,
                    "l"          : (_to_dict(node.leftchild)  if node.leftchild  is not None else None),
                    "r"          : (_to_dict(node.rightchild) if node.rightchild is not None else None)
                }
        if self.root is None:
            dict_repr = {}
        else:
            dict_repr = _to_dict(self.root)
        return json.dumps(dict_repr,indent=2)
    
    def split(self, n, height): 
        dimension = 0 
        if self.splitmethod == 'cycle':
            dimension = height % self.k 
        else: 
            spread_list = [] 
            for d in range(self.k):
                l = [] 
                for i in n.data: 
                    l.append(i.coords[d])
                spread_list.append(max(l) - min(l))
            spread_max = max(spread_list)
            dim = [] 
            for j in range(len(spread_list)):
                if spread_list[j] == spread_max: 
                    dim.append(j)
            dimension = min(dim)
        newdata = n.data.copy() 
        for a in range(len(newdata)): 
            for b in range(0,len(newdata)-a-1):
                c1 = [] 
                c2 = [] 
                for e in range(self.k):
                    dim2 = (dimension + e) % self.k 
                    c1.append(newdata[b].coords[dim2])
                    c2.append(newdata[b+1].coords[dim2])
                c1 = tuple(c1) 
                c2 = tuple(c2)
                if c1 > c2: 
                    newdata[b], newdata[b+1] = newdata[b+1], newdata[b]
        if len(newdata) % 2 == 0:
            median = float((newdata[len(newdata)//2].coords[dimension] + newdata[((len(newdata))//2) - 1].coords[dimension]) / 2)
        else: 
            median = float(newdata[len(newdata)//2].coords[dimension])
        lc = NodeLeaf(newdata[0:(len(newdata)//2)])
        rc = NodeLeaf(newdata[(len(newdata)//2):])
        return NodeInternal(dimension, median, lc, rc)

    def insert_helper(self, n, coord, code, height): 
        if isinstance(n, NodeLeaf): 
            n.data.append(Datum(coord, code))
            if len(n.data) > self.m: 
                return self.split(n,height)
            else: 
                return n 
        else: 
            if self.splitmethod == "cycle": 
                tempd = height % self.k 
                if n.splitindex != tempd: 
                    while(height % self.k) != n.splitindex: 
                        height += 1
            if coord[n.splitindex] < n.splitvalue: 
                n.leftchild = self.insert_helper(n.leftchild,coord,code,height+1)
            else:
                n.rightchild = self.insert_helper(n.rightchild, coord, code, height + 1)
            return n
        
    # Insert the Datum with the given code and coords into the tree.
    # The Datum with the given coords is guaranteed to not be in the tree.
    def insert(self,point:tuple[int],code:str):
        if self.root == None: 
            self.root = NodeLeaf([Datum(point,code)])
        else: 
            self.root = self.insert_helper(self.root,point,code,0)

    # Delete the Datum with the given point from the tree.
    # The Datum with the given point is guaranteed to be in the tree.

    def search(self, n, point): 
        if isinstance(n, NodeLeaf): 
            newdata = []
            for d in n.data:
                if d.coords != point:
                    newdata.append(d)
            if len(newdata) == 0:
                return None 
            n.data = newdata 
            return n

        if point[n.splitindex] > n.splitvalue: 
                n.rightchild = self.search(n.rightchild, point)
        elif point[n.splitindex] < n.splitvalue:
                n.leftchild = self.search(n.leftchild, point)
        else: 
            n.leftchild = self.search(n.leftchild, point)
            n.rightchild = self.search(n.rightchild, point)

        
        if n.rightchild == None and n.leftchild == None: 
            return None 
        elif n.rightchild == None and n.leftchild != None: 
            return n.leftchild
        elif n.rightchild != None and n.leftchild == None: 
            return n.rightchild
            
        return n
    

    def delete(self,point:tuple[int]):
        self.root = self.search(self.root,point)

    def bounding_box(self,n): 
        if isinstance(n, NodeLeaf):
            minimum = [float('inf')] * self.k 
            maximum = [float('-inf')] * self.k 
            for i in n.data: 
                for j in range(self.k): 
                    minimum[j] = min(minimum[j], i.coords[j])
                    maximum[j] = max(maximum[j], i.coords[j])

            result = []
        
            for j in range(self.k): 
                result.append([minimum[j],maximum[j]])
            return result
        else:
            if n.leftchild != None: 
                left = self.bounding_box(n.leftchild)
            else:
                left = None 
            if n.rightchild != None:
                right = self.bounding_box(n.rightchild)
            else:
                right = None 
            if left == None: 
                return right 
            elif right == None: 
                return left 
            result = [] 
            for j in range(self.k):
                mini = min(left[j][0], right[j][0])
                maxi = max(left[j][1], right[j][1])
                result.append([mini,maxi])
            return result 
        
    def point_dist(self, point1, point2): 
        dist = 0 
        for i in range(self.k): 
            dist += (point1[i] - point2[i]) ** 2
        return dist 

    def bb_dist(self, point, bb): 
        dist = 0 
        for i in range(self.k):
            if point[i] < bb[i][0]: 
                dist += (bb[i][0] - point[i]) ** 2 
            elif point[i] > bb[i][1]: 
                dist += (point[i] - bb[i][1]) ** 2 
        return dist
    
    def knn_helper(self,n, k, point, leaveschecked, l): 
        if isinstance(n,NodeLeaf): 
            for i in n.data:
                # if the list is full already and the point is farther than anything there we can keep going 
                if len(l) >= k: 
                    farthest = self.point_dist(point, l[-1].coords)
                    if (self.point_dist(point, i.coords) > farthest) or (self.point_dist(point,i.coords) == farthest and i.code > l[-1].code):
                        continue 
                index = len(l)
                for j in range(len(l)):
                    if (self.point_dist(point,i.coords) < self.point_dist(point, l[j].coords)) or (self.point_dist(point,i.coords) == self.point_dist(point,l[j].coords) and i.code < l[j].code):
                        index = j 
                        break 
                l.insert(index,i)
                # this loop can increase list more so we need an additional check
                if len(l) > k: 
                    l.pop()
            return 1
        if n.leftchild != None:
            lbb = self.bounding_box(n.leftchild)
        else: 
            lbb = None 
        if n.rightchild !=None : 
            rbb = self.bounding_box(n.rightchild)
        else:
            rbb = None 
        
        if lbb != None and rbb != None: 
            child = [(self.bb_dist(point,lbb),n.leftchild), (self.bb_dist(point,rbb),n.rightchild)]
        elif lbb != None and rbb == None: 
            child = [(self.bb_dist(point,lbb),n.leftchild), (float('inf'),n.rightchild)]
        elif lbb == None and rbb != None: 
            child = [(float('inf'), n.leftchild), (self.bb_dist(point,rbb),n.rightchild)]
        else: 
            child = [(float('inf'), n.leftchild), (float('inf'),n.rightchild)]
        count = 0
        if child[1][0] < child[0][0]:
            child[0], child[1] = child[1], child[0] 
        for d, c in child: 
            if (c != None) and (len(l) < k or d <= self.point_dist(point,l[-1].coords)):
                count += self.knn_helper(c,k,point,leaveschecked,l)
        return count 
    
    def knnquery(self,k:int,point:tuple[int]) -> str:
        # The next two lines just make it run.
        leaveschecked = 0
        knnlist = []
        if self.root != None:
            leaveschecked = self.knn_helper(self.root, k, point, leaveschecked, knnlist)
        return(json.dumps({"leaveschecked":leaveschecked,"points":[str({'coords': datum.coords,'code': datum.code}) for datum in knnlist]},indent=2))

    def range_helper(self, querybox,n):
        if isinstance(n,NodeLeaf):
            l = [] 
            for i in n.data: 
                boolean = True 
                index = 0 
                for j in range(self.k): 
                    if(i.coords[j] < querybox[index] or i.coords[j] > querybox[index + 1]): 
                        boolean = False 
                        break 
                    index += 2 
                if boolean == True: 
                    l.append(i)
            return l, 1
        else: 
           left = self.bounding_box(n.leftchild)
           right = self.bounding_box(n.rightchild)
           loverlap = 1
           index = 0 
           has_left = True
           for i in range(self.k):
               overlap1 = min(left[i][1], querybox[index+1]) - max(left[i][0], querybox[index])
               if overlap1 < 0: 
                   loverlap = 0
                   has_left = False 
                   break 
               loverlap *= overlap1
               index += 2
           roverlap = 1
           index = 0 
           has_right = True
           for i in range(self.k):
               overlap2 = min(right[i][1], querybox[index+1]) - max(right[i][0], querybox[index])
               if overlap2 < 0: 
                   roverlap = 0 
                   has_right = False 
                   break 
               roverlap *= overlap2
               index += 2
           result = []
           count = 0 
           if loverlap >= roverlap:
               if has_left:
                   left_result, left_count = self.range_helper(querybox,n.leftchild)
                   result.extend(left_result)
                   count += left_count
               if has_right:
                   right_result, right_count = self.range_helper(querybox,n.rightchild)
                   result.extend(right_result)
                   count += right_count
           elif roverlap > loverlap:
               if has_right:
                   right_result, right_count = self.range_helper(querybox,n.rightchild)
                   result.extend(right_result)
                   count += right_count
               if has_left: 
                   left_result, left_count = self.range_helper(querybox,n.leftchild)
                   result.extend(left_result)
                   count += left_count
           return result, count
        
    # Find all points in the querybox.
    def rangequery(self,querybox:List) -> str:
        # The next two lines just make it run.
        leaveschecked = 0
        rangelist = []

        rangelist, leaveschecked = self.range_helper(querybox,self.root)
        sorted = [] 
        for r in rangelist:
            sorted.append((r.code,r))
        sorted.sort()
        rangelist = []
        for i,j in sorted:
            rangelist.append(j)
        
        return(json.dumps({"leaveschecked":leaveschecked,"points":[str({'coords': datum.coords,'code': datum.code}) for datum in rangelist]},indent=2))
    




