import streamlit as st
from backbone import NAS
import pandas as pd
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from base import main as runner
import numpy as np

def search(self, num_iterations, pop_size = 100):
	pop = self.random_pop_gen(pop_size)
	bar = tqdm(range(num_iterations))
	st_bar = st.progress(0)
	progress = []
	for i in bar:
		pop,loss,acc = self.next_generation_generator(pop)
		progress.append([i+1, loss, acc])
		if(self.classification):
			bar.set_description(str({"i":i,"gen_loss":round(loss,3),"gen_acc":round(acc,3)}))
		else:
			bar.set_description(str({"i":i,"gen_loss":round(loss,3)}))
		st.progress((i+1)/num_iterations)
	pop,_,_ = self.sorter(pop)
	optimal_hyperparam = pop[0]
	N,m,c,activation = optimal_hyperparam
	loss,acc,model = runner(self.data, self.target_var, self.classification, self.lr, N, m, c, activation)
	print(N,m,c,activation,loss,acc)
	return model, progress, N, m, c, activation, loss, acc

NAS.search = search

def app():
	with st.sidebar:
		st.title("TabNAS")
		st.write("A Neural Network model generalized for any tabular dataset (Classification/Regression Task)")

		pop_size = st.slider("Set generational population size", min_value=12, max_value=200, step=4)
		num_iter = st.slider("Set number of optimization steps", min_value=5, max_value=100, step=1)

	uploaded_file = st.file_uploader("Upload dataset", type='csv')
	if uploaded_file is not None:
		df = pd.read_csv(uploaded_file)
		df = df.dropna(axis = 'columns')
		target_var = st.selectbox("Select target variable", list(df.columns))
		classification = st.selectbox("Select whether type is Classification or Regression", ["Classification", "Regression"])
		lr = float(st.text_input("Learning Rate", value="0.001"))
		classification = classification=="Classification"
		if st.button("Submit for NAS"):
			nas = NAS(df, target_var, classification, lr)
			model, progress, N, m, c, activation, loss, acc = nas.search(num_iter,pop_size)
			N = int(N)
			m = int(m)
			c = int(c)
			st.markdown('```py\nconfig = {\n"number_of_layers":'+str(N)+'\n"num_neurons":'+str([int(m/i+c) for i in range(2,N)])+'\n"activation":'+activation+'}```')
			if classification:
				performance = {"loss": round(loss,3), "acc": round(acc,3)}
			else:
				performance = {"loss": round(loss,3)}
			st.success("Performance"+str(performance))
			torch.save(model, "model.pt")
			with open("model.pt", "rb") as f:
				st.download_button('Download Model', f.read())
			progress = np.array(progress)
			if classification:
				fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,6))
				ax1.plot(progress.T[0]+1, progress.T[1], label="loss")
				ax1.set_xlabel("Gen No.")
				ax1.set_ylabel("Loss")
				ax1.legend()
				ax2.plot(progress.T[0], progress.T[2], label="acc")
				ax2.set_xlabel("Gen No.")
				ax2.set_ylabel("Acc")
				ax2.legend()
			else:
				fig, ax = plt.subplots()
				ax.plot(progress.T[0], progress.T[1], label="loss")
				ax.set_xlabel("Gen No.")
				ax.set_ylabel("Loss")
				ax.legend()
			st.pyplot(fig)


if __name__ == '__main__':
	app()