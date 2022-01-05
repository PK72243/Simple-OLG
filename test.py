###OLG
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy import interpolate
import time
import matplotlib.pyplot as plt

def my_lin(lb, ub, steps, spacing=1):
    span = (ub-lb)
    dx = 1.0 / (steps-1)
    return np.array([lb + (i*dx)**spacing*span for i in range(steps)])


class model_olg():
	def __init__(self,model_params,wage=1,interest=0.04):
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
		
	
	def crra_utility(self,c,l):
		if c<=0:
			return -9999999
		else:
			if self.sigma==1:
				return np.log(c)-self.kappa*((l**(1+1/self.v))/(1+1/self.v))
			else:
				return c**(1-self.sigma)/(1-self.sigma)-self.kappa*((l**(1+1/self.v))/(1+1/self.v))
	def model_utility(self,c,l):
		return getattr(self, self.utility_f)(c,l)
		
	def solve_hh(self,grid_params):
		
		self.a_min=grid_params['a_min']
		self.a_max=grid_params['a_max']
		self.grid_a_spacing=grid_params['grid_a_spacing']
		self.grid_a_size=int(grid_params['grid_a_size'])
		self.grid_a=my_lin(self.a_min,self.a_max,self.grid_a_size,self.grid_a_spacing)
		
		
		start=time.time()
		V=np.zeros((self.grid_a_size,self.N_income_levels))
		self.V_approx=dict()
		g_c=np.zeros((self.grid_a_size,self.N_income_levels))
		self.g_c_approx=dict()
		g_l=np.zeros((self.grid_a_size,self.N_income_levels))
		self.g_l_approx=dict()
		g_a=np.zeros((self.grid_a_size,self.N_income_levels))
		self.g_a_approx=dict()		
				
		for t in reversed(range(self.T_start,self.T_end+1)):
			self.V_approx[t]=dict()
			self.g_c_approx[t]=dict()
			self.g_a_approx[t]=dict()
			self.g_l_approx[t]=dict()
			
			for y in range(self.N_income_levels):

				for a in range(self.grid_a_size):
					if t<self.retirement_age:#Working
						def criter_func(params):
							c=params[0]
							l=params[1]
							V_=0
							if t<self.T_end:
								a_=max(self.a_min,min(self.w*self.income_levels[y]*params[1]+self.grid_a[a]*(1+self.r)-params[0],self.a_max))
								for k in range(self.N_income_levels):
									V_+=self.beta*self.V_approx[t+1][k](a_)*self.Pi[y,k]
							return -self.model_utility(c, l)-V_
					
						linear_cons=LinearConstraint([[1,0],[0,1],[-1,self.w*self.income_levels[y]]],[0,0,-self.grid_a[a]*(1+self.r)],[np.inf,1,np.inf])
						if t==self.T_end:
							x0=self.grid_a[a]+0.00001,0.5
							
						else:
							x0=[g_c[a,y],g_l[a,y]]
						res = minimize(criter_func, x0, method='trust-constr',constraints=[linear_cons],   options={'verbose': 0})

					else:#pension and not working
						def criter_func(params):
							c=params[0]
		
							V_=0
							if t<self.T_end:
								a_=max(self.a_min,min(self.grid_a[a]*(1+self.r)-params[0],self.a_max))
								for k in range(self.N_income_levels):
									V_+=self.beta*self.V_approx[t+1][k](a_)*self.Pi[y,k]
							return -self.model_utility(c, 0)-V_
						ub=self.grid_a[a]*(1+self.r)
						linear_cons=LinearConstraint([1],[0],[ub])
						if t==self.T_end:
							x0=[max(self.grid_a[a],ub/2)]
							
						else:
							x0=[max(g_c[a,y],ub/2)]
							
						if self.grid_a[a]<=0:
							res=minimize(criter_func,x0)
							res.fun=99999
							res.x=[0,0]
						else:

							res = minimize(criter_func, x0, method='trust-constr',constraints=[linear_cons],   options={'verbose': 0})

						
					V[a,y]=-res.fun
					g_c[a,y]=res.x[0]
					if t<self.retirement_age:
						g_l[a,y]=res.x[1]
						g_a[a,y]=self.w*self.income_levels[y]*res.x[1]+self.grid_a[a]*(1+self.r)-res.x[0]
					else:
						g_l[a,y]=0
						g_a[a,y]=self.grid_a[a]*(1+self.r)-res.x[0]
	
				self.V_approx[t][y]=interpolate.interp1d(self.grid_a,V[:,y])
				self.g_a_approx[t][y]=interpolate.interp1d(self.grid_a,g_a[:,y])
				self.g_c_approx[t][y]=interpolate.interp1d(self.grid_a,g_c[:,y])
				self.g_l_approx[t][y]=interpolate.interp1d(self.grid_a,g_l[:,y])
					
			print(f'Generation {t} solved')
		print(f'Household problems solved in {round(time.time()-start,2)} seconds')
		
	def lifetime_sim(self,n,initial_prob,a_0):
				
		states=np.empty((n,self.N_periods))
		states[:,0]=np.random.choice(np.arange(self.N_income_levels),p=initial_prob,size=n)
		
		
		for t in range(1,self.N_periods):
			for s in range(self.N_income_levels):
				states[:,t][states[:,t-1]==s]=np.random.choice(np.arange(self.N_income_levels),p=self.Pi[s],size=np.sum(states[:,t-1]==s))
				
		a=np.empty((n,self.N_periods+1))	
		c=np.empty((n,self.N_periods))
		l=np.empty((n,self.N_periods))
		V=np.empty((n,self.N_periods))
		a[:,0]=0
		
			
		for t in range(self.N_periods):
			for s in range(self.N_income_levels):
				idx_temp=states[:,t]==s
				a_=np.maximum(self.a_min,np.minimum(a[idx_temp,t],self.a_max))	
				a[idx_temp,t+1]=self.g_a_approx[t+self.T_start][s](a_)
				c[idx_temp,t]=self.g_c_approx[t+self.T_start][s](a_)
				l[idx_temp,t]=self.g_l_approx[t+self.T_start][s](a_)
				V[idx_temp,t]=self.V_approx[t+self.T_start][s](a_)
				
		return V,a,c,l
		
		
params={'kappa':5.24,'beta':0.988,'v':2,'sigma':1.5,'omega':1,'r':0.04,
		'Pi':np.array([[0.7,0.3],[0.3,0.7]]),'y_levels':np.array([0.8,1.2]),'utility_f':'crra_utility',
		'T_start':24,'T_end':80,'retirement':True,'retirement_age':65}
grid_params={'a_min':0,'a_max':3,'grid_a_spacing':1.5,'grid_a_size':10}

aa=model_olg(params)
aa.solve_hh(grid_params)
n=10000
initial_prob=[0.5,0.5]
V,a,c,l=aa.lifetime_sim(n, initial_prob, 0)
