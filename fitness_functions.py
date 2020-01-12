import numpy as np
import scipy.stats as spstat

path = './supportData/'


#    o = np.array(list(map(float, f.readline().strip().split())))

def fitness(x, prob_num, o = None, M = None, A = None):
    def fitness1(x):
        """Shifted Sphere Function"""
        d = x.shape[0]
            
#        with open(path + 'sphere_func_data.txt', 'r') as f:
#            o = np.array(list(map(float, f.readline().strip().split())))[:d]
        z = x - o
    
        bias = -450
        f = np.sum(z**2) + bias
        return f
    
    def fitness2(x):
        """Shifted Schwefel's Problem 1.2"""
        d = x.shape[0]
    
#        with open(path + 'schwefel_102_data.txt', 'r') as f:
#            o = np.array(list(map(float,f.readline().strip().split())))[:d]
    
        z = x - o
        bias = -450
        f = 0
        for i in range(d):
            s = 0
            for j in range(i+1):
                s += z[j]
            f += s**2
        f += bias
        return f
    
    def fitness3(x):
        """Shifted rotated high conditioned elliptic"""
        d = x.shape[0]
    
#        with open(path + 'high_cond_elliptic_rot_data.txt', 'r') as f:
#            o = np.array(list(map(float,f.readline().strip().split())))[:d]
#        with open(path + 'elliptic_M_D30.txt', 'r') as f:
#            M = np.zeros([d,d])
#            for i in range(d):
#                line = f.readline()
#                M[i,:] = np.array(list(map(float,line.strip().split())))[:d]
    
        f = 0
        bias = -450
    
        z = np.matmul(x-o, M)
    
        for i in range(d):
            f += (10**6)**(i/(d-1))*(z[i]**2)
        f += bias
        return f
    
    def fitness4(x):
        """Shifted Schwefel's Problem 1.2 with Noise in Fitness"""
        d = x.shape[0]
#        with open(path + 'schwefel_102_data.txt', 'r') as f:
#            o = np.array(list(map(float,f.readline().strip().split())))[:d]
    
        N = np.random.randn()
        z = x - o
        bias = -450
        f = 0
        for i in range(d):
            s = 0
            for j in range(i+1):
                s += z[j]
            f += s ** 2
        f *= (1+0.4*np.abs(N))
        f += bias
        return f
    
    def fitness5(x):
        """Schwefel's Problem 2.6 with Global Optimum on Bounds"""
        d = x.shape[0]
    
#        with open(path + 'schwefel_206_data.txt', 'r') as f:
#            o = np.array(list(map(float,f.readline().strip().split())))[:d]
#            A = np.zeros([d,d])
#            i = 0
#            for i in range(d):
#                line = f.readline()
#                A[i,:] = np.array(list(map(float, line.strip().split())))[:d]
    
        bias = -310
        B = np.matmul(A,o)
    
        f = np.max(np.abs(np.matmul(A, x) - B))
    
        f += bias
        return f
    
    def fitness6(x):
        """Shifted Rosenbrock's Function"""
        d = x.shape[0]
    
#        with open(path + 'rosenbrock_func_data.txt', 'r') as f:
#            o = np.array(list(map(float,f.readline().strip().split())))[:d]
    
        z = x - o + 1
        bias = 390
        f = 0
        for i in range(d-1):
            f += (100*(z[i]**2-z[i+1])**2 + (z[i]-1)**2)
    
        f += bias
        return f
    
    
    def fitness7(x):
        """Shifted Rotated Griewank's Function without Bounds"""
        d = x.shape[0]
    
#        with open(path + 'griewank_func_data.txt', 'r') as f:
#            o = np.array(list(map(float,f.readline().strip().split())))[:d]
#        with open(path + 'griewank_M_D30.txt', 'r') as f:
#            M = np.zeros([d,d])
#            for i in range(d):
#                line = f.readline()
#                M[i,:] = np.array(list(map(float,line.strip().split())))[:d]
    
        z = x - o
        z = np.matmul(x - o, M)
        bias = -180
        t = 0
        for i in range(d):
            t += z[i]**2
        t /=4000
    
        s = 1
        for i in range(d):
            s *= np.cos(z[i]/np.sqrt(i+1))
    
        f = t - s + 1
        f += bias
        return f
    
    def fitness8(x):
        """Shifted rotated Ackleys's Function with Global Optimum on Bounds"""
        d = x.shape[0]
#        with open(path + 'ackley_func_data.txt', 'r') as f:
#            o = np.array(list(map(float,f.readline().strip().split())))[:d]
#        with open(path + 'ackley_M_D30.txt', 'r') as f:
#            M = np.zeros([d,d])
#            for i in range(d):
#                line = f.readline()
#                M[i,:] = np.array(list(map(float,line.strip().split())))[:d]
    
        z = x - o
        z = np.matmul(x - o, M)
        f =0
        s = 0
        r = 0
        for i in range(d):
            s += z[i]**2
        for i in range(d):
            r += np.cos(2*np.pi*z[i])
        bias = -140
    
        f += -20*np.exp(-0.2*np.sqrt(s/d))
        f -= np.exp(r/d)
        f += 20 + np.exp(1)
        f += bias
        return f
    
    
    def fitness9(x):
        """Shifted Rastrigin Function"""
        d = x.shape[0]
    
#        with open(path + 'rastrigin_func_data.txt', 'r') as f:
#            o = np.array(list(map(float,f.readline().strip().split())))[:d]
    
        z = x - o
        bias = -330
        f = 0
        for i in range(d):
            f += z[i]**2 - 10*np.cos(2*np.pi*z[i])
        f += (10 * d)
        f += bias
        return f
    
    def fitness10(x):
        """Shifted Rotated Rastrigin Function"""
        d = x.shape[0]
    
#        with open(path + 'rastrigin_func_data.txt', 'r') as f:
#            o = np.array(list(map(float,f.readline().strip().split())))[:d]
#        with open(path + 'rastrigin_M_D30.txt', 'r') as f:
#            M = np.zeros([d,d])
#            for i in range(d):
#                line = f.readline()
#                M[i,:] = np.array(list(map(float,line.strip().split())))[:d]
    
        z = x - o
        z = np.matmul(x - o, M)
        bias = -330
        f = 0
        for i in range(d):
            f += z[i]**2 - 10*np.cos(2*np.pi*z[i])
        f += 10 * d
        f += bias
        return f
    
    if prob_num == 1:
        return fitness1(x)
    elif prob_num == 2:
        return fitness2(x)
    elif prob_num == 3:
        return fitness3(x)
    elif prob_num == 4:
        return fitness4(x)
    elif prob_num == 5:
        return fitness5(x)
    elif prob_num == 6:
        return fitness6(x)
    elif prob_num == 7:
        return fitness7(x)
    elif prob_num == 8:
        return fitness8(x)
    elif prob_num == 9:
        return fitness9(x)
    elif prob_num == 10:
        return fitness10(x)
    

if __name__ == "__main__":
    x = np.array([-40.57, 58.03, -45.57, -71.55, -19.75, -80.14, -7.70, 23.96, 92.74, 6.26, -9.65, \
                  -25.85, -10.71, 5.71, 71.66, 71.04, -59.73, 77.42, -68.63, 63.40, 32.08, -38.14, 28.74, \
                  -90.10, -79.95, -70.87, 42.69, 23.85, 30.35, 90.04])
    print(fitness1(x))