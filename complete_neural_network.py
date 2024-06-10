import math
import numpy as np 
import pandas as pd 
it=2
layers=5
neurons=[10,5,5,5,5,5,6]
x=[]
y=[]
x=np.random.rand(10,1)
y=np.random.rand(10,1)

def relu(z):
    return np.maximum(z,0)
def softmax(z):
    exps=np.exp(z)
    return exps/np.sum(exps)
def forward_prop(x,layers,neurons,w,b,y):
    x=np.random.rand(neurons[0],1)
    a=[x]
    z=[]
    for i in range(layers+1):
        z.append(w[i].dot(a[i])+b[i])
        if i!=layers:
            a.append(relu(w[i].dot(a[i])+b[i]))
        else:
            a.append(softmax(z[-1]))
    a[-1]=a[-1]/np.amax(a[-1])
    sub=np.subtract(a[-1],y)
    loss=np.square(sub)
    print("aaaaaaaaaaaaaaaaaa",np.subtract(a[-1],y))
    return loss,a,z
def deriv_relu(z):
    return z>0
def backward_prop(loss,a,w,b,z,neurons):
    dw={}
    db={}
    dz={}
    da={}
    da[str(len(neurons)-1)]=2*(np.sqrt(loss))
    for i in range(len(neurons)-1,-1,-1):
        if i!=0:
            derz=deriv_relu(z[i-1])
            dz[str(i-1)]=(1/neurons[i-1])*(da[str(i)].dot(deriv_relu(z[i-1]).T))
            dz[str(i-1)]=np.sum(dz[str(i-1)],axis=1)
            dz[str(i-1)]=dz[str(i-1)].reshape((len(dz[str(i-1)]),1))
            dw[str(i-1)]=(1/neurons[i-1])*dz[str(i-1)].dot(a[i-1].T)
            db[str(i-1)]=(1/neurons[i-1])*dz[str(i-1)]
            da[str(i-1)]=dz[str(i-1)].T.dot(w[i-1])
            da[str(i-1)]=da[str(i-1)].T
    return dw,db

def update_params(w,b,dw,db,a):
    for i in range(len(w)):
        w[i]=w[i]-(a*dw[str(i)])
        b[i]=b[i]-(a*db[str(i)])
    return w,b

def gradient_descent(it,x,y,layers,neurons):
    a1=0.05
    w=[]
    b=[]
    for i in range(layers+1):
        w.append(np.random.rand(neurons[i+1],neurons[i]))
        b.append(np.random.rand(neurons[i+1],1))
    for i in range(it):
        for i in range(layers):
            loss,a,z=forward_prop(x[i],layers,neurons,w,b,y[i])
            print("loss",loss)
            dw,db=backward_prop(loss,a,w,b,z,neurons)
            w,b=update_params(w,b,dw,db,a1)
            
gradient_descent(it,x,y,layers,neurons)