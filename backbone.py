'''
no. of layers
no of neurons in each layers - optimise m and c(change isnt too much (min no of neurons)) (y=m/x + c)
activation function - sigmoid,relu,tanh
'''
import numpy as np
from base import main as runner
from base import read_file
from tqdm import tqdm

class NAS:
	def __init__(self, data, target_var, classification, lr, N_min = 2, N_max = 10, m_min = 2, m_max = 20, c_min = 5, c_max = 20):
		self.data = data
		self.target_var = target_var
		self.classification = classification
		self.lr = lr
		self.ranges = {"N":list(range(N_min,N_max)),
						"m":list(range(m_min, m_max)),
						"c":list(range(c_min, c_max)),
						"activation":("sigmoid","tanh","ReLU")}

	def random_pop_gen(self,num):
		pop = []
		for i in range(num):
			N = np.random.choice(self.ranges['N'])
			m = np.random.choice(self.ranges['m'])
			c = np.random.choice(self.ranges['c'])
			activation = np.random.choice(self.ranges['activation'])
			row = [N,m,c,activation]
			pop.append(row)
		return pop

	def loss_single(self,row):
		loss,acc, _ = runner(self.data, self.target_var, self.classification, self.lr, row[0], row[1], row[2], row[3])
		return loss,acc

	def compute_loss(self,pop):
		loss_arr = []
		acc_arr = []
		for row in pop:
			loss,acc = self.loss_single(row)
			loss_arr.append(loss)
			acc_arr.append(acc)
		return loss_arr, sum(acc_arr)/len(acc_arr)

	def sorter(self,pop):
		loss_arr,acc = self.compute_loss(pop)
		loss_arr = np.array(loss_arr)
		sorted_loss_indices = np.argsort(loss_arr)
		pop = np.array(pop)
		pop = pop[sorted_loss_indices]
		return pop,np.mean(loss_arr),acc 

	def offspring(self,pop):
		parents = [[pop[i],pop[i+1]] for i in range(0,len(pop),2)]
		child_arr = []
		for p_i,p_j in parents:
			child = []
			for i,j in zip(p_i,p_j):
				if(np.random.random()>0.5):
					child.append(i)
				else:
					child.append(j)
			child_arr.append(child)
		return child_arr

	def next_generation_generator(self,pop):
		sorted_pop,loss,acc = self.sorter(pop)
		sorted_pop = sorted_pop[:len(sorted_pop)//2]
		offspring_pop = self.offspring(sorted_pop)
		random_pop = self.random_pop_gen(len(offspring_pop))
		new_pop = [i for i in sorted_pop]+[i for i in offspring_pop]+[i for i in random_pop]
		return new_pop,loss,acc

	def search(self, num_iterations, pop_size = 100):
		pop = self.random_pop_gen(pop_size)
		bar = tqdm(range(num_iterations))
		progress = []
		for i in bar:
			pop,loss,acc = self.next_generation_generator(pop)
			progress.append([i+1, loss, acc])
			if(self.classification):
				bar.set_description(str({"i":i,"gen_loss":round(loss,3),"gen_acc":round(acc,3)}))
			else:
				bar.set_description(str({"i":i,"gen_loss":round(loss,3)}))
		pop,_,_ = self.sorter(pop)
		optimal_hyperparam = pop[0]
		N,m,c,activation = optimal_hyperparam
		loss,acc,model = runner(self.data, self.target_var, self.classification, self.lr, N, m, c, activation)
		print(N,m,c,activation,loss,acc)
		return model, progress, N, m, c, activation, loss, acc

if __name__ == '__main__':
	data = read_file("./data/adult.csv")
	target_var = 'income'
	classification = True
	lr = 0.001

	nas = NAS(data, target_var, classification, lr)
	model = nas.search(10,16)