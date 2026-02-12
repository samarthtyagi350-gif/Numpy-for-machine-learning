import numpy as np

class ArrayFundamentals:
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def get_x(self):
        return self.x,self.x.shape
    
    def get_y(self):
        return self.y,self.y.shape
        
        






x=np.array([ [1200, 2],[1500, 3],[1800, 4]])

y = np.array([200000, 250000, 300000])


s=ArrayFundamentals(x,y)

a1,a2=s.get_x()
b1,b2=s.get_y()
print(f"feature matrix {a1}\n of shape {a2}")
print(f"target vector {b1} \nof shape {b2}")