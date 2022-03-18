import sys
sys.path.append('lsh_imt/')
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
import time
import random
from math import e
from lsh import LSH
from matrix_simhash import SimHash


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.zeros_(m.weight)

def data_vector_transform(vector,input_size,label):
    dim=input_size
    transformed_vector=torch.zeros(dim+dim+1)
    for i in range(dim):
        transformed_vector[i]=vector[i]*vector[i]
    for k in range(dim):
        transformed_vector[k+dim]=vector[k]*label
    transformed_vector[dim+dim]=label*label
    return transformed_vector
def query_transform(weight,weight_star,input_size,learning_rate):
    dim=input_size
    transformed_weight=torch.zeros(1,dim+dim+1)
    for i in range(dim):
        transformed_weight[0,i]=(learning_rate-1)*weight[i]*weight[i]+weight[i]*weight_star[i]
    transformed_weight[0,dim:dim+dim]=(1-2*learning_rate)*weight.t()-weight_star.t()
    transformed_weight[0,dim+dim]=learning_rate
    return transformed_weight

def data_wrapping(dataset,input_size):
	data_size=len(dataset)
	data_dim=input_size
	wrapped_data=torch.zeros(data_size,data_dim+data_dim+1)
	for i in range(data_size):
		data_vector,data_label=dataset[i]
		wrapped_data[i]=data_vector_transform(data_vector,input_size,data_label)
	print("data wrapped")
	return wrapped_data



def hashing(wrap_set,K,L):
	wrap_size,dimension=wrap_set.size()
	lsh = LSH(SimHash(dimension, K, L),wrap_size, K, L)
	lsh.clear()
	lsh.insert_multi(wrap_set, wrap_size)

	print("data indexed")
	return lsh


def total_loss(model,train_set,criterion):
    train_size=len(train_set)
    new_loss=0
    for i in range(train_size):
        data,label = train_set[i]
        pred=model(data)
        new_loss+=criterion(pred, label)
    return new_loss/train_size




data_path='data/space_ga.npy'
data=np.load(data_path)
cut_size=int(0.7*len(data))

train_data,test_data=data[:cut_size],data[cut_size:]
Xtrain,Ytrain=train_data[:,:-1],train_data[:,-1]
Xtest,Ytest=test_data[:,:-1],test_data[:,-1]
input_size=Xtrain.shape[1]

x=torch.from_numpy(Xtrain).float().to(device)
x=F.normalize(x, dim=1, p=2)
y=torch.from_numpy(Ytrain).float().to(device)
y=torch.unsqueeze(y,1)
train_set=torch.utils.data.TensorDataset(x,y)

x_test=torch.from_numpy(Xtest).float().to(device)
x_test=F.normalize(x_test, dim=1, p=2)
y_test=torch.from_numpy(Ytest).float().to(device)
y_test=torch.unsqueeze(y_test,1)
test_set=torch.utils.data.TensorDataset(x_test,y_test)


# Hyper-parameters 
num_classes=1
max_iter_teacher=2000
max_iter=2000
learning_rate=0.01
tol=1e-15
K=9
L=30000
# Regression model
model = nn.Sequential(
          nn.Linear(input_size, num_classes,bias=False),
        )

model=model.to(device)

# Loss and optimizer
criterion = nn.MSELoss() 
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)







print("table (K,L): "+str(K)+' '+str(L))
wrapped_set=data_wrapping(train_set,input_size)
wrapped_set=wrapped_set.to(device)
lsh=hashing(wrapped_set,K,L)

model.load_state_dict(torch.load('model.ckpt', map_location="cuda:0"))
w_batch=model[0].weight.data.detach().clone()
w_batch=w_batch.t()

# approximate teaching
model.apply(init_weights)
iteration=0
loss_list=[]
iter_list=[]
time_list=[]
test_list=[]

total_time=0
reg_lambda=0
train_size=len(train_set)

loss_w=torch.zeros(train_size,1).to(device)
id_list_tmp=[]
select_id_tmp=0
while True:

    if iteration%100==0:
        new_loss=total_loss(model,train_set,criterion)
        test_loss=total_loss(model,test_set,criterion)
        print('Step {}, Train Loss: {:.4f}, Test Loss: {:.4f}, Total Time: {:.4f}'.format(iteration, new_loss.item(),test_loss.item(),total_time))
        iter_list.append(iteration)
        loss_list.append(new_loss.item())
        time_list.append(total_time)
        test_list.append(test_loss.item())

    start=time.time()
    w=model[0].weight.t().data
    query_vector=query_transform(w,w_batch,input_size,learning_rate)
    
    query_vector=-query_vector.to(device)
    id_list=lsh.query_multi(query_vector, 1)

    select_id=id_list[0]

    images,labels=train_set[select_id]
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    end=time.time()
    total_time+=end-start
    iteration+=1
    if iteration>max_iter or loss.item()<tol:
        break

