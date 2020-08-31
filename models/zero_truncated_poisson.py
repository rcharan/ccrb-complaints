from scipy.optimize import newton_krylov as newton
from numpy import exp, log
import numpy as np

class ZeroTruncatedPoisson:

    def __init__(self, lamda = None):
      self.lamda = lamda
      self.data = None
    
    def fit(self, data):
      '''Fit to count data
      
      Parameters
      ----------
      
      Data: a numpy Series with counts
      '''
      
      self.data  = data
      self.x_bar = data.mean()
      self.n     = len(data)
      
      def mle_eqn(lamda):
        return lamda / (1 - exp(-lamda)) - self.x_bar
      
      [self.lamda] = newton(mle_eqn, [self.x_bar])
      
    def is_fit(self):
      return self.data is not None
      
    def log_pmf(self, k):
      terms = (
        [-log(m) for m in range(1, k+1)] # 1/k!
       +[k * log(self.lamda), -log(exp(self.lamda) - 1)]
      )
      
      return sum(terms)
      
    def pmf(self, k):
      return exp(self.log_pmf(k))
    
    def log_likelihood(self, data = None):
      '''Log likelihood on a per-observation basis'''
      
      if data is None:
        data = self.data
      
      return self.data.map(self.log_pmf).mean()
    
    def plot(self, max_count = None, include_zero = False, normalize = False):
      '''Returns x and y coordinates for plotting the pmf.
      
      Parameters
      ----------
      
      max_count: maximum value of x to return. If None, the maximum
                 value in the datset is used
                 
      include_zero: whether to include 0 in the xs
      
      normalize: whether to return a PMF or expected counts (using
                 the size of the data)
      
      '''        
      if max_count is None:
        max_count = self.data.max()
        
      xs = np.arange(start = 0 if include_zero else 1,
                     stop  = max_count + 1)
      
      pmf = np.vectorize(self.pmf)
      
      ys = pmf(xs)
      
      if not normalize:
        ys *= self.n
      
      return xs, ys