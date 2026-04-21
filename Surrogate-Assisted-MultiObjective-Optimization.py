import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

x1=np.random.uniform(10,50,1000)
x2=np.random.uniform(100,200,1000)
x3=np.random.uniform(1,5,1000)

Q=x1*x2*np.log(x3+1)
P=(x1**2/(x3+1))+0.1*x2

Dataframe=pd.DataFrame({'Flowrate' : x1,
                        'Inlet_Temperature': x2,
                        'Length': x3,
                        'Heat_Transfer': Q,
                        'PressureDrop': P})

print(Dataframe.head())

X=Dataframe[['Flowrate','Inlet_Temperature','Length']].values
Y_Q=Dataframe['Heat_Transfer'].values
Y_P=Dataframe['PressureDrop'].values

model_Q=RandomForestRegressor(n_estimators=100,random_state=42)
model_P=RandomForestRegressor(n_estimators=100,random_state=42)
model_Q.fit(X,Y_Q)
model_P.fit(X,Y_P)

print('The AI machine successfully trained!')

print(f'Accuracy of Heat Transfer prediction is : {model_Q.score(X,Y_Q)*100:.2f} %')
print(f'Accuracy of Pressure Drop prediction is : {model_P.score(X,Y_P)*100:.2f} %')

class ThermalOptimization(Problem):
    def __init__(self):
        super().__init__(n_var=3,n_obj=2,n_ieq_constr=0,
                        xl=np.array([10,100,1]),
                        xu=np.array([50,200,5]))
    def _evaluate(self,x,out,*args,**kwargs):
        pred_Q=model_Q.predict(x)
        pred_P=model_P.predict(x)        
        out['F']=np.column_stack([-pred_Q,pred_P])

problem=ThermalOptimization()

algorithm=NSGA2(pop_size=100)

print('Optimization process has been started!')
Opt=minimize(problem,algorithm,('n_gen',50),seed=1,verbose=False)
print('Optimization seccessfully finished!')

Optimal_Heat_Transfer=-Opt.F[:,0]
Optimal_Presure_Drop=Opt.F[:,1]

max_allowable_P = 200
valid_indices=Optimal_Presure_Drop<=max_allowable_P
valid_Q=Optimal_Heat_Transfer[valid_indices]
valid_P=Optimal_Presure_Drop[valid_indices]

best_idx=np.argmax(valid_Q)
golden_Q=valid_Q[best_idx]
golden_P=valid_P[best_idx]

print("\n--- Golden Design Point ---")
print(f"Maximum Heat Transfer under constraints: {golden_Q:.2f}")
print(f"Pressure Drop: {golden_P:.2f}")

plt.figure(figsize=(8,6))
plt.scatter(Optimal_Heat_Transfer,Optimal_Presure_Drop,color='red',marker='o')
plt.title('Pareto front: Heat Transfer vs Pressure Drop')
plt.xlabel('Heat Transfer (Maximized)')
plt.ylabel('Pressure Drop (Minimized)')
plt.grid(True)
plt.show()

