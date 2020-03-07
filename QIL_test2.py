import torch
import torch.nn as nn
import numpy as np
import copy

'''
목표 MobileNet v2에 빠르게 QIL 적용

1. alexnet training  <- success
2. apply QIL to alexnet (= reproduce) 
   - know How to quantize ex ) aware quantization - > pytorch source
   - know How to apply QIL on alexnet
   - conduct training
3. Apply QIL to MobileNet v2
'''

class Quantizer():
    
    discretization_level = 3
    def __init__(self):
        torch.manual_seed(0)
        init_num = torch.randn(5) - 0.5
        self.cw = torch.tensor(init_num[0],requires_grad = True)
        self.dw = torch.tensor(init_num[1],requires_grad = True)
        self.cx = torch.tensor(init_num[2],requires_grad = True)
        self.dx = torch.tensor(init_num[3],requires_grad = True)
        self.gamma = torch.tensor(init_num[4],requires_grad = True) 

    def quantize(self, target, param):
        if target is 'weight':
            self.transfomer_weights(param)
        elif target is 'activation':
            self.transfomer_activation(param)
        else:
            print("Warning : Target is not weight or activation")
        self.discretizer(param)

    def transfomer_weights(self,weights):
        aw,bw = (0.5 / self.dw) , (-0.5*self.cw / self.dw + 0.5)       
        
        weights[ abs(weights) < self.cw - self.dw] = 0
        weights = torch.where(  abs(weights) > self.cw + self.dw, 
                                weights.sign(), weights)
        weights = torch.where( (abs(weights) >= self.cw - self.dw) & (abs(weights) <= self.cw + self.dw),
                                (aw*abs(weights) + bw)**self.gamma * weights.sign() , weights)

    def transfomer_activation(self,x):
        ax,bx = (0.5 / self.dx) , (-0.5*self.cx / self.dx + 0.5)       
        
        x[ x < self.cx - self.dx] = 0
        x[ x > self.cx + self.dx] = 1 
        x = torch.where( (abs(x) >= self.cx - self.dx) & (abs(x) <= self.cx + self.dx),
                            ax*abs(x) + bx, x)
    
    def discretizer(self,tensor):
        q_D = pow(2, Quantizer.discretization_level)
        torch.round_(tensor.mul_(q_D))
        tensor.div_(q_D)

class QILconv2d(nn.Conv2d):
    pass


if __name__ == '__main__':
    inputs = torch.rand(1,5,2,2).sub_(0.5).mul_(5)
    conv1 = nn.Conv2d(5,3,1,bias = False)
    q = Quantizer()
    print("before : ",conv1.weight)
    q.quantize('weight', conv1.weight)
    print("after : ",conv1.weight)
    
    out = conv1(inputs)
    
    #weights2 = copy.deepcopy(weights1) + 1 -1
    
    #print("before weights 1\n",weights1)
    #print("before weights 2\n",weights2)
    
    
   