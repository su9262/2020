import torch
import numpy as np

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
    
    discretization_level = 32
    def __init__(self):
        torch.manual_seed(0)
        self.cw,self.dw,self.cx,self.dx,self.gamma = torch.randn(5) - 0.5
        print(self.cw,self.dw,self.cx,self.dx,self.gamma)

    def quantize(self, target, param):
        if target is 'weight':
            self.transfomer(param,self.cw,self.dw,self.gamma)
        elif target is 'activation':
            self.transfomer(param,self.cx,self.dx)
        else:
            print("Warning : Target is not weight or activation")
        self.discretizer(param)

    def transfomer(self,tensor,c_delta,d_delta,r = 1):
        outplane,inplane,kh,kw = tensor.size()
        for o in range(outplane):
            for i in range(inplane):
                for h in range(kh):
                    for w in range(kw):
                        t = tensor[o][i][h][w]
                        if abs(t) < c_delta - d_delta:
                            t = 0
                        elif abs(t) > c_delta + d_delta:
                            t.sign_()
                        else:
                            a,b = (0.5 / d_delta) , (-0.5*c_delta / d_delta + 0.5)
                            t = (a*abs(w) + b)**r * t.sign()
                        tensor[o][i][h][w] = t
    
    def discretizer(self,tensor):
        q_D = pow(2, Quantizer.discretization_level)
        torch.round_(tensor.mul_(q_D))
        tensor.div_(q_D)


if __name__ == '__main__':
    weights = torch.rand(2,2,2,2).sub_(0.5)
    activation = torch.rand(4,4,4,4)
    q = Quantizer()
    print("before \n",weights)
    q.quantize('weight', weights)
    #q.transfomer_weights(param,c_delta,d_delta,r)
    #q.discretizer(weights)
    print("after \n",weights)

    
