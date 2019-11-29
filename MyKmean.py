import numpy as np
import cv2

class Kmean:
	def __init__(self , cluster_num , input_dim):
		self.cluster_num = cluster_num

		self.C =[]# 300*np.random.random([cluster_num , input_dim])
		# for i in range(cluster_num):
		# 	self.C.append(np.array([np.random.randint(255), np.random.randint(255), np.random.randint(255)]))
		# self.C = np.array(self.C)
		self.input_dim = input_dim
		self.clusters = []

	def setRand(self):
		for i in range(self.cluster_num):
			self.C.append(self.input_data[np.random.randint(self.input_data.shape[0])])

	def setClusterNum(self,cluster_num):
		self.cluster_num = cluster_num;
		self.__init__(self.cluster_num, self.input_dim)

	def setInputDimension(self , input_dim):
		self.input_dim = input_dim 
		self.__init__(self.cluster_num , input_dim)

	def dist(self,p1,p2):
		return np.sqrt(np.sum((p2-p1)**2))

	def centerOfMass(self , data_arr):
		s = 0;
		n = 0;
		# print(data_arr)
		for i in data_arr:
			s = s + np.float64(i)
			n = n+1;
		# print(s)
		if (n==0):
			return np.array([-1000,-1000,-1000])
		return (1.0/n)*s;

	def train(self, input_data , iter_num = 10):
		self.input_data = input_data
		self.setRand()
		for i in range(iter_num):
			self.train_(input_data)

	def train_(self , input_data):
		self.lastClusters = self.clusters.copy
		self.clusters = [[] for i in range(self.cluster_num)];
		for data in input_data:
			d , last_d = 0 , 100000000;
			nearest_ci = 0;
			for ci in range(self.cluster_num):
				d = self.dist(data, self.C[ci])
				if d < last_d:
					last_d = d ;
					nearest_ci = ci ;
			self.clusters[nearest_ci].append(data)

		self.clusters = np.array(self.clusters)
		for ci in range(self.cluster_num): 
			if len(self.clusters[ci]) > 0:
				self.C[ci] = self.centerOfMass(self.clusters[ci])

	def getModifiedData(self):
		output_data = [];
		for data in self.input_data:
			d , last_d = 0 , 100000000;
			nearest_ci = 0;
			for ci in range(self.cluster_num):
				d = self.dist(data, self.C[ci])
				if d < last_d:
					last_d = d ;
					nearest_ci = ci ;
			output_data.append(self.C[nearest_ci])
		return output_data


	def getCenterOfCluster(self):
		return self.C


if __name__ == "__main__":
	kmean = Kmean(5,3);
	kmean.train(img_vec)
	ut_img = np.uint8(kmean.getModifiedData())


