###OLG
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy import interpolate
import time

def my_lin(lb, ub, steps, spacing=1):
    span = (ub-lb)
    dx = 1.0 / (steps-1)
    return np.array([lb + (i*dx)**spacing*span for i in range(steps)])


class model_olg():
	def __init__(self,model_params,grid_params,wage=1,interest=0.04):
		self.kappa=float(model_params['kappa'])
		self.beta=float(model_params['beta'])
		self.sigma=float(model_params['sigma'])
		self.v=float(model_params['v'])
		self.Pi=model_params['Pi']
		self.income_levels=model_params['y_levels']
		self.N_income_levels=self.Pi.shape[0]
		
		self.w=float(wage)
		self.r=float(interest)
		
		self.utility_f=model_params['utility_f']
		
		self.T_start=int(model_params['T_start'])
		self.T_end=int(model_params['T_end'])
		self.N_periods=self.T_end-self.T_start+1
		self.retirement=model_params['retirement']
		if self.retirement==True:
			self.retirement_age=int(model_params['retirement_age'])
		else:
			self.retirement_age=self.T_end
		
		self.a_0=model_params['a_0']
		self.a_min=grid_params['a_min']
		self.a_max=grid_params['a_max']
		self.grid_a_spacing=grid_params['grid_a_spacing']
		self.grid_a_size=int(grid_params['grid_a_size'])
		self.grid_a=my_lin(self.a_min,self.a_max,self.grid_a_size,self.grid_a_spacing)
		
	
	def crra_utility(self,c,l):
		if self.sigma==1:
			return np.log(c)-self.kappa*((l**(1+1/self.v))/(1+1/self.v))
		else:
			return c**(1-self.sigma)/(1-self.sigma)-self.kappa*((l**(1+1/self.v))/(1+1/self.v))
	def model_utility(self,c,l):
		return getattr(self, self.utility_f)(c,l)
		
	def solve_hh(self):
		
		start=time.time()
		self.V=np.empty((self.grid_a_size,self.N_income_levels,self.N_periods))
		self.g_c=np.empty((self.grid_a_size,self.N_income_levels,self.N_periods))
		self.g_l=np.empty((self.grid_a_size,self.N_income_levels,self.N_periods))
		self.g_a=np.empty((self.grid_a_size,self.N_income_levels,self.N_periods))
				
				
		for t in reversed(range(self.T_start,self.T_end+1)):
			
			for y in range(self.N_income_levels):
				if t<self.T_end:
					interpoland={}
					for k in range(self.N_income_levels):
						interpoland[k]=interpolate.interp1d(self.grid_a,V[:,k,t-self.T_start+1])
			
				for a in range(self.grid_a_size):
					if t<self.retirement_age:#Working
						def criter_func(params):
							c=params[0]
							l=params[1]
							V_=0
							if t<self.T_end:
								a_=max(self.a_min,min(self.w*self.income_levels[y]*params[1]+self.grid_a[a]*(1+self.r)-params[0],self.a_max))
								for k in range(self.N_income_levels):
									V_+=self.beta*interpoland[k](a_)*self.Pi[y,k]
							return -self.model_utility(c, l)-V_
					
						linear_cons=LinearConstraint([[1,0],[0,1],[-1,self.w*self.income_levels[y]]],[0,0,-self.grid_a[a]*(1+self.r)],[np.inf,1,np.inf])
						if t==self.T_end:
							x0=self.grid_a[a]+0.00001,0.5
							
						else:
							x0=[self.g_c[a,y,t+1-self.T_start],self.g_l[a,y,t+1-self.T_start]]
					else:#pension and not working
						def criter_func(params):
							c=params[0]
		
							V_=0
							if t<self.T_end:
								a_=max(self.a_min,min(self.grid_a[a]*(1+self.r)-params[0],self.a_max))
								for k in range(self.N_income_levels):
									V_+=self.beta*interpoland[k](a_)*self.Pi[y,k]
							return -self.model_utility(c, 0)-V_
					
						linear_cons=LinearConstraint([1],[0],[self.grid_a[a]*(1+self.r)])
						if t==self.T_end:
							x0=[self.grid_a[a]+0.00001]
							
						else:
							x0=self.g_c[a,y,t+1-self.T_start]
					try:
						res = minimize(criter_func, x0, method='trust-constr',constraints=[linear_cons],   options={'verbose': 0})
					except:
						res=minimize(criter_func,x0)
						res.fun=99999
						res.x=[0,0]
					self.V[a,y,t-self.T_start]=-res.fun
					self.g_c[a,y,t-self.T_start]=res.x[0]
					if t<self.retirement_age:
						self.g_l[a,y,t-self.T_start]=res.x[1]
						self.g_a[a,y,t-self.T_start]=self.w*self.income_levels[y]*res.x[1]+self.grid_a[a]*(1+self.r)-res.x[0]
					else:
						self.g_l[a,y,t-self.T_start]=0
						self.g_a[a,y,t-self.T_start]=self.grid_a[a]*(1+self.r)-res.x[0]
		
		
					
			print(t)
		print(time.time()-start)
		

		
		
		
		
params={'kappa':5.24,'beta':0.988,'v':2,'sigma':1,'a_0':0,'omega':1,'r':0.04,
		'Pi':np.array([[0.7,0.3],[0.3,0.7]]),'y_levels':np.array([0.8,1.2]),'utility_f':'crra_utility',
		'T_start':24,'T_end':80,'retirement':True,'retirement_age':65}
grid_params={'a_min':0.05,'a_max':3,'grid_a_spacing':1.5,'grid_a_size':10}

aa=model_olg(params,grid_params).solve_hh()
