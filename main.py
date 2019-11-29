from MyKmean import Kmean

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

class Gaussian:
    def __init__(self ,x_mean , sigma):
        self.mean = x_mean
        self.sigma = sigma
    
    def dist(self , p1 , p2):
        dd = np.array(p1) - np.array(p2)
        return np. sqrt(dd[0]**2 + dd[1]**2)

    def eval(self, x):
        aa =  pow((1/(self.sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*(self.dist(x,self.mean))/(self.sigma)),2);
        # print(aa)
        return aa


class perceptron:
    def __init__(self , m):
        m = m+1
        # self.threshold = np.random.random()
        self.w =  np.random.random(m)
        self.eta = .25
        self.Err = 0

    def eval(self,x):
        x = np.array(x)
        x = np.append(x , -1)
        net = np.sum((x*self.w))
        return net

    def train__(self , x , d):
        x = np.array(x)
        y = np.append(x , -1)
        o = self.eval(x)
        self.w = self.w + 0.5*self.eta*(d - o)*y##*(1 - o**2)
        self.Err = self.Err + (d-o)**2;
        return self.Err

    def train(self , input_x , input_y , n):
        Emax = 3.1
        while(1):
            # print("1111")
            err = 0
            for i in range(n):
                x = input_x[i]
                y = input_y[i]
                err = self.train__(x,y)
            
            print(err)
            if err < Emax:
                break
            else:
                self.Error_reset()

    def Error_reset(self):
        self.Err = 0


class RBF:
    def __init__(self, RBF_Func_number ):
        self.RBF_Func_number = RBF_Func_number
        self.kmean = Kmean(RBF_Func_number,RBF_Func_number)
        self.m = RBF_Func_number
        self.perceptron = perceptron(self.m)
        # self.bias = np.random.random()

    def fit(self , input_data):
        self.kmean.train(input_data , iter_num = 10)
        self.centerOfClusters = self.kmean.getCenterOfCluster()
        print(self.centerOfClusters)
        c = self.centerOfClusters
        diff  = c[1]-c[0]
        d = np.sqrt((diff[0]**2 + diff[1]**2))

        # d = abs(max(self.centerOfClusters) - min(self.centerOfClusters))
        self.sigma = d/(np.sqrt(2*self.m))
        self.GaussianFunctions = []
        self.centerOfClusters = np.array([[3,2] , [8,7] , [9,3] , [2,8]])
        for c in self.centerOfClusters:
            self.GaussianFunctions.append(Gaussian(c , self.sigma))

    def train(self , x , d):
        input_data_new = []
        # print(x)
        for input_data in x:
            x_new = []
            # print(input_data[0])
            a1 = self.GaussianFunctions[0].eval(input_data)
            a2 = self.GaussianFunctions[1].eval(input_data)
            a3 = self.GaussianFunctions[2].eval(input_data)
            a4 = self.GaussianFunctions[3].eval(input_data)
            # a5 = self.GaussianFunctions[4].eval(input_data)
            
            # for i in range(self.m):
            #     # F = F + self.GaussianFunctions[i](self.w[i] * self.input_data[i]) 
                # x_new.append(self.GaussianFunctions[i].eval(input_data[i]))
            input_data_new.append([a1,a2,a3,a4])
            # input_data_new.append([a1,a2,a3,a4,a5])
            # print([a1,a2])
        # print(input_data_new)

        x = np.array(x)
        self.perceptron.train(input_data_new , d, x.shape[0])

    def eval(self , data):
        F = []
        for input_data in data:
            a1 = self.GaussianFunctions[0].eval(input_data)
            a2 = self.GaussianFunctions[1].eval(input_data)
            a3 = self.GaussianFunctions[2].eval(input_data)
            a4 = self.GaussianFunctions[3].eval(input_data)
            x_new = [a1,a2,a3,a4]
            f = self.perceptron.eval(x_new)    
               
            F.append(f)
        return np.array(F)

if __name__=="__main__":
    print("RBF ...")
    x1 = [[1,8],[2,9],[3,8],[2,7],[3,6],[4,8],[6,4],[7,1],[7,3],[7,4],[7,5],[8,1],[8,2],[8,3],[9,1],[9,2],[9,4],[10,2],[10,3],[10,4]]
    d1 = [1]*20
    x2 = [[1,1],[1,3],[2,1],[2,2],[3,1],[3,2],[3,3],[4,2],[4,3],[4,4],[5,4],[5,5],[6,7],[7,6],[8,8],[8,9],[9,7],[9,8],[9,9],[10,10]]
    d2 = [0]*20

    x = x1+x2
    d = d1+d2
    x = np.array(x)
    d = np.array(d)
    rbf = RBF(4)
    rbf.fit(x)
    rbf.train(x,d)
    print("end")
    print("x = [1.1,8.1]" + str(rbf.eval([[1.1,8.1]]))) ## 1
    print("x = [3.1,2.5]" + str(rbf.eval([[13.1,2.5]]))) ## 0



    # # Make data.
    # X = np.arange(0, 9, 0.5)
    # Y = np.arange(0, 9, 0.5)
    # xx = []
    # yy = []
    # zz = []

    # for i in X:
    #     for j in Y:
    #         xx.append(i)
    #         yy.append(j)
    #         zz.append(rbf.eval([[i,j]]))

    # xx = np.array(xx)
    # yy = np.array(yy)
    # zz = np.array(zz)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    # # Plot the surface.
    # ax.plot_wireframe(xx, yy, zz, rstride=10, cstride=10)

    # plt.show()