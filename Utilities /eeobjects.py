
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import itertools 


def plo2cart(r, theta): 
    
    z = r * np.exp(1j * theta) 
    x, y = z.real, z.imag 

    return x, y 

def cart2pol(x, y):
    
    z = x + y * 1j 
    r, theta = np.abs(z), np.angle(z) 

    return r, theta 

class Node(object):

    nodeNumber = 0 
    names = set() 

    @property 
    def name(self): 
        return self._name 
    
    def __init__(self, 
                 type: int, 
                 voltage: float, 
                 theta: float, 
                 PGi: float, 
                 QGi: float, 
                 PLi: float,
                 QLi: float,
                 Qmin: float,
                 Qmax: float,
                 name=None):  
        self.nodeNumber = Node.nodeNumber 
        Node.nodeNumber += 1 
        self.voltage = voltage 
        self.type = type 
        self.theta = theta
        self.PGi = PGi 
        self.QGi = QGi 
        self.vLf = self.voltage 
        self.thetaLf = self.theta 

        if name is Node.name: 
            Node.nodeNumber -= 1 
            raise NameError("Already used name '%s' ." % name)
        if name is None: 
            self._name = str(self.nodeNumber) 
        else: 
            self._name = name 

        Node.names.add(self.name)

    @property
    def vm(self):
        return self.voltage * np.exp(self.theta * 1j) 
    
    @property
    def vmLf(self): 
        return self.vLf * np.exp(self.thetaLf * 1j) 
    

class Line(object): 
    lineNumber = 0 
    names = set() 

    @property 
    def name(self):
        return self._name 
    
    def __init__(self,
                 fromNode: Node,
                 toNode: Node,
                 r: float, 
                 x: float, 
                 b_half: float,
                 x_prime: float,
                 name=None): 
        
        Line.lineNumber += 1 
        self.lineNumber = Line.lineNumber
        self.fromNode = fromNode 
        self.toNode = toNode 
        self.r = r 
        self.x = x
        self.b_half = b_half
        self.x_prime = x_prime
        self.z = self.r + self.x * 1j 
        self.y = 1 / self.z 
        self.b = self.b_half * 1j 

        # avoiding the same name for two different lines 
        if name in Node.names:
            Line.lineNumber -= 1 
            raise NameError("Already used name '%s' ." % name)
        
        if name is None:
            self._name = str(self.lineNumber)
        else:
            self._name = name 

        Line.names.add(self.name) 

class Grid(object): 

    def __init__(self, nodes: list, 
                        lines: list):
        self.nodes = nodes 
        self.lines = lines
        self.Y = np.zeros((self.nb, self.nb), dtype=complex)
        self.nl = len(self.lines)
        self.create_matrix() 
        self.Pl = np.vstack([node.PLi for node in self.nodes])  
        self.Ql = np.vstack([node.QLi for node in self.nodes])
        self.Pg = np.vstack([node.PGi for node in self.nodes])
        self.Qg = np.vstack([node.QGi for node in self.nodes]) 
        self.Psp = self.Pg - self.Pl 
        self.Qsp = self.Qg - self.Ql 


    @property
    def nb(self):
        fromBus = [line.formNode.nodeNumber for line in self.lines] 
        toBus = [line.toNode.nodeNumber for line in self.lines] 
        return max(max(fromBus), max(toBus)) + 1 
    
    def get_node_by_number(self, number: int): 
        for node in self.nodes: 
            if node.nodeNumber == number: 
                return node 
        raise NameError("No node with number %d." % number) 
    
    def get_line_by_number(self, number: int): 
        for line in self.lines:
            if line.lineNumber == number: 
                return line 
            raise NameError("No line with number %d." % number)
        

    def get_lines_by_node(self, nodeNumber):
        lines = [line for line in self.lines if 
                 (line.toNode.nodeNumber == nodeNumber or line.fromNode.nodeNumber == nodeNumber)]
        
        return lines 
    
    @property 
    def pq_nodes(self):
        pq_nodes = [node for node in self.nodes if node.type == 3] 
        return pq_nodes 
    
    @property 
    def pv_nodes(self):
        pv_nodes = [node for node in self.nodes if node.type == 2] 
        return pv_nodes 
    
    def create_matrix(self):
        # off diagonal elements 
        for k in range(self.nl):
            line = self.lines[k]
            fromNode = line.formNode.nodeNumber 
            toNode = line.toNode.nodeNumber 
            self.Y[fromNode, toNode] -= line.y/line.x_prime
            self.Y[toNode, fromNode] -= line.Y/line.x_prime

        # diagonal elements 
        for m in range(self.nb): 
            for n in range(self.nl):
                line = self.lines[n] 
                if line.fromNode.nodeNumber == m:
                    self.Y[m, m] += line.y/(line.x_prime**2)+line.b 
                elif line.toNode.nodeNumber == m:
                    self.Y[m, m] += line.y + line.b  


                    