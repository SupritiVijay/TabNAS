import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

def read_file(file_path):
	df = pd.read_csv(file_path, nrows=1000)
	df = df.dropna(axis = 'columns')
	return df

class Preprocessor:
	def __init__(self, cat_range=10):
		self.cat_range = cat_range
		self.properties = {}

	def preprocess(self, df):
		num_features = len(df.columns)-1
		for col_name in df.columns:
			col = df[col_name]
			unique_val = np.unique(col).tolist()
			col_range = len(unique_val)
			if(type(col[0]) == str or col_range<self.cat_range):
				self.properties[col_name]={'categorical':unique_val}
			else:
				stand_deviation = np.std(col)
				mean = np.mean(col)
				self.properties[col_name]={'numerical':{'mean':mean,'standard_deviation':stand_deviation}}

	def transform(self,df):
		data = {}
		for col_name in self.properties.keys():
			if(list(self.properties[col_name].keys())[0]=='categorical'):
				vocab = self.properties[col_name]['categorical']
				col_values = []
				for row in df[col_name]:
					col_values.append(vocab.index(row))
				data[col_name]=col_values
			else:
				local_prop = self.properties[col_name]['numerical']
				mean = local_prop['mean']
				stand_deviation = local_prop['standard_deviation']
				col_values = []
				for row in df[col_name]:
					col_values.append((row - mean)/stand_deviation)
				data[col_name]=col_values

		data = pd.DataFrame(data)
		data.columns = self.properties.keys()
		return data

class Model(torch.nn.Module):
	def __init__(self, N, m , c, activation, properties,target_var, sorted_columns, embedding_dim = 10, classification=True, device=None):
		super(Model, self).__init__()
		self.device = device
		self.properties = properties
		N = int(N)
		m = int(m)
		c = int(c)
		if(classification):
			self.target_count = len(self.properties[target_var]['categorical'])
		else:
			self.target_count = 1
		del self.properties[target_var]
		if activation=='sigmoid':
			self.activation = torch.nn.Sigmoid()
		elif activation=='tanh':
			self.activation = torch.nn.Tanh()
		else:
			self.activation = torch.nn.ReLU()
		self.count_num_features = 0
		self.count_categorical_features = 0
		self.categorical_features_embedding = []
		self.sorted_columns = sorted_columns[:]
		del self.sorted_columns[sorted_columns.index(target_var)]
		self.target_var = target_var
		self.embedding_dim = embedding_dim
		self.preset()
		self.input_linear = torch.nn.Linear(self.count_num_features+(self.count_categorical_features*self.embedding_dim), int(m + c))
		self.intermediate_linear_neurons = [int(m/i+c) for i in range(2,N)]
		self.intermediate_linear = []
		prev_neurons = int(m/1 + c)
		for n in  self.intermediate_linear_neurons:
			layer = torch.nn.Linear(prev_neurons,n)
			if self.device is not None:
				layer = layer.to(self.device)
			self.intermediate_linear.append(layer)
			prev_neurons = n
		self.output_linear = torch.nn.Linear(int(m/(N-1) + c),self.target_count)

	def preset(self):
		for col_name in self.sorted_columns:
			if(list(self.properties[col_name].keys())[0]=='categorical'):
				vocab = self.properties[col_name]['categorical']
				self.count_categorical_features +=1
				embedding = torch.nn.Embedding(num_embeddings = len(vocab), embedding_dim = self.embedding_dim)
				if self.device is not None:
					embedding = embedding.to(self.device)
				self.categorical_features_embedding.append(embedding)
			else:
				self.count_num_features += 1

	def forward(self, x_num, x_cat):
		x = x_num
		for index,col in enumerate(x_cat.transpose(0,1)):
			embedding_layer = self.categorical_features_embedding[index]
			embedding = embedding_layer(col)
			x = torch.cat((x,embedding), 1)
		x = self.input_linear(x)
		for layer in self.intermediate_linear:
			x = self.activation(x)
			x = layer(x)
		x = self.activation(x)
		x = self.output_linear(x)
		return x

class TableDataset(torch.utils.data.Dataset):
	def __init__(self, properties, data, target_var, classification = True):
		self.properties = properties
		self.columns = list(data.columns)
		self.cat_columns = []
		self.num_columns = []
		self.data = data.values
		self.target_var = target_var
		self.classification=classification
		self.preset()

	def preset(self):
		for col_name in self.columns:
			if(list(self.properties[col_name].keys())[0]=='categorical'):
				self.cat_columns.append(col_name)
			else:
				self.num_columns.append(col_name)

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, idx):
		row = self.data[idx]
		x_cat = []
		x_num = []
		if(self.classification):
			y = int(row[self.columns.index(self.target_var)])
		else:
			y = row[self.columns.index(self.target_var)]
			y = torch.Tensor([float(y)])
		for v,col_name in zip(row,self.columns):
			if(col_name==self.target_var):
				continue
			if(col_name in self.num_columns):
				x_num.append(v)
			else:
				x_cat.append(v)
		x_cat = torch.Tensor(x_cat)
		x_cat = x_cat.to(torch.long)
		x_num = torch.Tensor(x_num)
		x_num = x_num.to(torch.float32)
		return x_cat,x_num,y

def main(data, target_var, classification, lr, N, m, c, activation):
	preprocessor = Preprocessor()
	preprocessor.preprocess(data)
	data_transformed = preprocessor.transform(data)
	dataset = TableDataset(preprocessor.properties, data_transformed, target_var, classification = classification)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size = 16, shuffle = True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = Model(N, m , c, activation, preprocessor.properties, target_var, dataset.columns, classification = classification, device=device)
	model.to(device)
	if(classification):
		criterion = torch.nn.CrossEntropyLoss()
	else:
		criterion = torch.nn.MSELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
	for epoch in range(10):
		running_loss =0.0
		running_acc = 0.0
		bar = tqdm(dataloader,disable = True)
		for i,(batch_x_cat, batch_x_num, batch_y) in  enumerate(bar):
			batch_x_num = batch_x_num.to(device)
			batch_x_cat = batch_x_cat.to(device)
			batch_y = batch_y.to(device)
			optimizer.zero_grad()
			out = model(batch_x_num,batch_x_cat)
			loss = criterion(out, batch_y)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			if(classification):
				pred = torch.argmax(out, dim = 1)
				acc = pred == batch_y
				acc = torch.mean(acc.float())
				running_acc += acc.item()
				bar.set_description(str({"epoch":epoch,"loss":round(running_loss/(i+1),3),"acc":round(running_acc/(i+1),3)}))
			else:
				bar.set_description(str({"epoch":epoch,"loss":round(running_loss/(i+1),3)}))
		bar.close()
	return running_loss/(i+1),running_acc/(i+1), model

if __name__ == '__main__':
	data = read_file("adult.csv")
	target_var = 'income'
	classification = True
	lr = 0.001
	N = 2
	m = 10
	c = 6
	activation = 'ReLU'
	loss = main(data,target_var,classification,lr,N, m, c, activation)

	