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
    
    discretization_level = 3
    def __init__(self):
        torch.manual_seed(0)
        self.cw,self.dw,self.cx,self.dx,self.gamma = torch.randn(5) - 0.5
        print(self.cw,self.dw,self.cx,self.dx,self.gamma)

    def quantize(self, target, param):
        if target is 'weight':
            self.transfomer_weights(param)
        elif target is 'activation':
            self.transfomer_activation(param)
        else:
            print("Warning : Target is not weight or activation")
        self.discretizer(param)

    def transfomer_weights(self,weights):
        outplane,inplane,kh,kl = weights.size()
        for o in range(outplane):
            for i in range(inplane):
                for h in range(kh):
                    for l in range(kl):
                        w = weights[o][i][h][l]
                        if abs(w) < self.cw - self.dw:
                            w = 0
                        elif abs(w) > self.cw + self.dw:
                            w.sign_()
                        else:
                            aw,bw = (0.5 / self.dw) , (-0.5*self.cw / self.dw + 0.5)
                            w = (aw*abs(w) + bw)**self.gamma * w.sign()
                        weights[o][i][h][l] = w

    def transfomer_activation(self,x):
        outplane,inplane,kh,kl = x.size()
        for o in range(outplane):
            for i in range(inplane):
                for h in range(kh):
                    for l in range(kl):
                        _x = x[o][i][h][l]
                        if _x < self.cx - self.dx:
                            _x = 0
                        elif _x > self.cx + self.dx:
                            _x = 1
                        else:
                            ax,bx = (0.5 / self.dx) , (-0.5*self.cx / self.dx + 0.5)
                            _x = ax*_x + bx
                        x[o][i][h][l] = _x
    
    def discretizer(self,tensor):
        q_D = pow(2, Quantizer.discretization_level)
        torch.round_(tensor.mul_(q_D))
        tensor.div_(q_D)


if __name__ == '__main__':
    weights = torch.rand(2,2,2,2).sub_(0.5).mul_(5)
    activation = torch.rand(4,4,4,4)
    q = Quantizer()
    print("before \n",weights)
    q.quantize('weight', weights)
    #q.transfomer_weights(param,c_delta,d_delta,r)
    #q.discretizer(weights)
    print("after \n",weights)

    
