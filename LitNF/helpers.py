import torch
from scipy import interpolate
from scipy.stats import norm
import numpy as np
from torch import optim
from nflows.distributions.base import Distribution


"""Note that this is not used because its bad"""
class StandardNormalTemp(Distribution):
        """A multivariate Normal with zero mean and a covariance that 
            can be chosen to be any value.
            From images generation it resulted that a lower variance gives 
            better sample result - this did not show the same effect here"""
        def __init__(self, shape,temp=1):
            super().__init__()
            self._shape = torch.Size(shape)
            self.temp=temp
            self.register_buffer("_log_z",
                                 torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi),
                                              dtype=torch.float64),
                                 persistent=False)

        def _log_prob(self, inputs, context):
            # Note: the context is ignored.
            if inputs.shape[1:] != self._shape:
                raise ValueError(
                    "Expected input of shape {}, got {}".format(
                        self._shape, inputs.shape[1:]
                    )
                )
            neg_energy = -0.5 * \
                torchutils.sum_except_batch(inputs ** 2, num_batch_dims=1)
            return neg_energy - self._log_z

        def _sample(self, num_samples, context):
            if context is None:
                return torch.normal(std=torch.ones(self._shape, device=self._log_z.device)*self.temp)
            else:
                # The value of the context is ignored, only its size and device are taken into account.
                context_size = context.shape[0]
                samples = torch.randn(context_size * num_samples, *self._shape,
                                      device=context.device)
                return torchutils.split_leading_dim(samples, [context_size, num_samples])

        def _mean(self, context):
            if context is None:
                return self._log_z.new_zeros(self._shape)
            else:
                # The value of the context is ignored, only its size is taken into account.
                return context.new_zeros(context.shape[0], *self._shape)




def mass(data,canonical=False):
    
    if canonical:
        n_dim=data.shape[1]
        p=data.reshape(-1,n_dim//3,3)
        px=p[:,:,0]
        py=p[:,:,1]
        pz=p[:,:,2]
        
        E=torch.sqrt(px**2+py**2+pz**2)
        E=E.sum(axis=1)**2
        p=px.sum(axis=1)**2+py.sum(axis=1)**2+pz.sum(axis=1)**2
        # return torch.sqrt(E-p) 

        return torch.sqrt(torch.max(E-p,torch.zeros(len(E)).to(E.device))) 
    else:
        n_dim=data.shape[1]
        p=data.reshape(-1,n_dim//3,3)
        px=torch.cos(p[:,:,1])*p[:,:,2]
        py=torch.sin(p[:,:,1])*p[:,:,2]
        pz=torch.sinh(p[:,:,0])*p[:,:,2]
        E=torch.sqrt(px**2+py**2+pz**2)
        E=E.sum(axis=1)**2
        p=px.sum(axis=1)**2+py.sum(axis=1)**2+pz.sum(axis=1)**2
        m2=E-p
        
        assert m2.isnan().sum()==0  
        return torch.sqrt(torch.max(m2,torch.zeros(len(E)).to(E.device))) 
        
def preprocess(data,rev=False):
        n_dim=data.shape[1]
        data=data.reshape(-1,n_dim//3,3)
        p=torch.zeros_like(data)
        if rev:
            p[:,:,0]=torch.arctanh(data[:,:,2]/torch.sqrt(data[:,:,0]**2+data[:,:,1]**2+data[:,:,2]**2))
            p[:,:,1]=torch.atan2(data[:,:,1],data[:,:,0])
            p[:,:,2]=torch.sqrt(data[:,:,0]**2+data[:,:,1]**2)
            
        else:
            
            p[:,:,0]=data[:,:,2]*torch.cos(data[:,:,1])
            p[:,:,1]=data[:,:,2]*torch.sin(data[:,:,1])
            p[:,:,2]=data[:,:,2]*torch.sinh(data[:,:,0])
        return p.reshape(-1,n_dim)
def F(x): #in: 1d array, out: functions transforming array to gauss
        print(len(x))
        ix= np.argsort(x)
        y=np.linspace(0,1,len(ix))
        x=x[ix]+np.random.rand(len(x))*0.01*np.random.rand(len(x))
        x=np.sort(x)
        
        fun=interpolate.PchipInterpolator(x,y)
        funinv=interpolate.PchipInterpolator(y,x)
        return fun,funinv
def marginal_flows(data):
        
        f,ffi=F(data)
        fi=lambda x: fbar(ffi,x,min(x),max(x))
        return f,fi
def fbar(f,x,minx,maxx):
        xbar=f(x)+0
        xbar[x<minx]=f(x[x<minx])*np.exp(-abs(x[x<minx]-minx))
        xbar[x>maxx]=f(x[x>maxx])*np.exp(-abs(maxx-x[x>maxx]))
        return xbar
def mmd(x, y, kernel="rbf"):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
      

    return torch.mean(XX + YY - 2. * XY)
class Rational(torch.nn.Module):
    """Rational Activation function.
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are initialized to the best rational 
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """
    def __init__(self):
        super().__init__()
        self.coeffs = torch.nn.Parameter(torch.Tensor(4, 2))
        self.reset_parameters()

    def reset_parameters(self):
        self.coeffs.data = torch.Tensor([[1.1915, 0.0],
                                    [1.5957, 2.383],
                                    [0.5, 0.0],
                                    [0.0218, 1.0]])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0,1].zero_()
        exp = torch.tensor([3., 2., 1., 0.], device=input.device, dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        output = torch.div(PQ[..., 0], PQ[..., 1])
        return output
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor