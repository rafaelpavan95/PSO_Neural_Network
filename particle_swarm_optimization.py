import numpy as np

def PSO_optimizer(X_train, X_test, y_train, y_test, neural_network, c1=2, c2=2, wmin=0.4, wmax=0.9, max_iter = 100, n_particles = 100, verbose=True):

    # Weights initialization (normal distribution mean 0 [-1,1])
    # Each line is a particle. Each particle is a vector of weights and biases
    # The entire matrix is the swarm
    
    dim = neural_network.n_input*neural_network.n_neurons_hl1 + neural_network.n_neurons_hl1*neural_network.n_neurons_hl2 + neural_network.n_neurons_hl2*neural_network.n_output + neural_network.n_output  + neural_network.n_neurons_hl1 + neural_network.n_neurons_hl2
    weights = np.random.randn(n_particles, dim)

    v0 = np.random.randn(n_particles, dim)*0.1
    
    gbest_list = []

    gbest_train_val = []

    gbest_test_val = []

    acc_train = []

    acc_test = []
    

    for iter in range(1,max_iter+1):

        fitness_train = []

        fitness_test = []
            
        # fitness evaluation for each particle of the swarm

        for p in range(n_particles):

            fitness_train.append(neural_network.feedforward(weights[p,:], X_train, y_train))
            
            fitness_test.append(neural_network.feedforward(weights[p,:], X_test, y_test))
        
        fitness_train = np.array(fitness_train)
        
        fitness_test = np.array(fitness_test)

        # update of pbest matrix (cognitive component)

        if iter == 1:

            pbest = weights.copy()
            pbest_fitness = fitness_train.copy()
            position_gbest = np.argmin(pbest_fitness)
            gbest = pbest[position_gbest,:].copy()


        else:

            for p in range(n_particles):

                if fitness_train[p]<pbest_fitness[p]:

                    pbest[p,:] = weights[p,:].copy()
                    pbest_fitness[p] = fitness_train[p].copy()

                else: pass

        
        # update gbest

        position_gbest = np.argmin(pbest_fitness)

        new_gbest = pbest[position_gbest,:]
        
        if neural_network.feedforward(gbest,X_train,y_train) > neural_network.feedforward(new_gbest,X_train,y_train):
            gbest = new_gbest.copy()

        else: pass

        gbest_list.append(gbest)

        gbest_train_val.append(neural_network.feedforward(gbest,X_train,y_train))
        
        gbest_test_val.append(neural_network.feedforward(gbest,X_test,y_test))

        # update inertia weight (linearly decreasing)

        inertia_weight = wmax - (wmax-wmin)*iter/max_iter

        # update velocity and positions

        v = v0*inertia_weight + c1*np.random.rand(n_particles, dim)*(pbest-weights) + c2*np.random.randn(n_particles, dim)*(gbest-weights)
        
        v[v>1]=1
        v[v<-1]=-1

        v0 = v.copy()

        
        # update positions

        weights = weights + v
        
        train_probs = neural_network.predictions(gbest_list[-1], X_train)
        test_probs = neural_network.predictions(gbest_list[-1], X_test)

        if verbose==True:
            if iter%25 == 0:
                print("\n----------")
                print(f'Iteration {iter}')
                print(f"Train Fitness (Loss+Regularization): {gbest_train_val[-1]}")
                print(f"Test Fitness (Loss+Regularization): {gbest_test_val[-1]}")
                print(f"Train Accuracy: {np.sum(y_train == np.argmax(train_probs,axis=1))/len(y_train)*100}")
                print(f"Test Accuracy: {np.sum(y_test == np.argmax(test_probs,axis=1))/len(y_test)*100}")
                
            else:pass
        else: pass

        acc_train.append(np.sum(y_train == np.argmax(train_probs,axis=1))/len(y_train)*100)
        acc_test.append(np.sum(y_test == np.argmax(test_probs,axis=1))/len(y_test)*100)

    results=    {
                
                'train_acc': acc_train,
                'test_acc': acc_test,
                'list_gbest': gbest_list,
                'train_loss': gbest_train_val,
                'test_loss': gbest_test_val
                }

    return results