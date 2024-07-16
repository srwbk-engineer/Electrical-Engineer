
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

        
    
                 
        