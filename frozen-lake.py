

### Package Imports ###
from os import system
from time import sleep
import gym 
import numpy as np


State = int
Action = int

class DynamicProgramming:
    """
    """

    def __init__(self, env, gamma=0.95, epsilon=0.001):
        '''
        Initialize policy, environment, value table (V), policy (policy), and transition matrix (P)
        '''
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.policy = self.create_initial_policy()
        self.V = np.zeros(self.num_states) # Vector of estimated utilities for each state, states are numbered 0-15, initialized at zero
        self.P = self.env.P # Dict of a Dict of list of tuples. Outer dict keys = states, 0-15, index into a state, you get a dict of 4 actions that can be performed, for each action we have
        # a list of possible states that you can end up in, then it's a tuple that tells us what prob of ending up in each of those states, the state indexer, the reward of doing that, finally
        # a boolean indicator of if its a terminal state or not
        self.gamma = gamma # Discount rate
        self.epsilon = epsilon # Convergence parameter
        self.rewards = {state:0 for state in range(self.num_states)} # The reward function R(s)
        self.terminal_states = {state:False for state in range(self.num_states)} # Returns a True or False indicating if the game state provided 
        # is in a terminal state (i.e. no further actions can be taken)
        
        for state,actions in env.P.items():
            for action,action_data in actions.items():
                prob, state, r, is_end = action_data[0]
                if is_end==True:
                    self.terminal_states[state] = True # Record if terminal state (ice hole or goal)
                    if r == 1:
                        self.rewards[state] = 1 # If a goal state, then R(s) = 1, else R(s) left at 0
                
    def create_initial_policy(self):
        '''
        A policy is a numpy array of length self.num_states where
        self.policy[state] = action

        '''
        # policy is num_states array (deterministic)
        policy = np.zeros(self.num_states, dtype=int)
        return policy

    def updated_action_values(self, state: State) -> np.ndarray:
        """
        This is a useful helper function for implementing value_iteration.
        Given a state (given by index), returns a numpy array of entries
        
        Z_i = SUM[p(s'|s,a_i)*U(s')] over all s' for action a_i
        
        i.e. return a np.array: [Z_1, ..., Z_n]
        
        based on current value function self.V for U(s')
        """
        
        actionvalues = np.zeros(self.num_actions)
        
        for action in range(self.num_actions):
            zisum = 0
            for tup in self.P[state][action]:
                zisum += tup[0]*self.V[tup[1]]
            actionvalues[action] = zisum
        
        return actionvalues


    def value_iteration(self):
        """
        Perform value iteration to compute the value of every state under the optimal policy.
        This method does not return anything. After calling this method, self.V should contain the
        correct values for each state. Additionally, self.policy should be an array that contains
        the optimal policy, where policies are encoded as indicated in the `create_initial_policy` docstring.
        """

        changesinu = np.zeros(self.num_states)

        stoppingcondition = False
        stoppingconditionvalue = (self.epsilon*(1-float(self.gamma)))/self.gamma

        while (not stoppingcondition):
            for i in range(self.num_states):
                actionvalues = self.updated_action_values(i)

                maxvalueaction = 0.0
                indexOfMaxValueAction = 0
                for j in range(len(actionvalues)):
                    if (actionvalues[j] > maxvalueaction):
                        maxvalueaction = actionvalues[j]
                        indexOfMaxValueAction=j

                if (self.terminal_states[i]):
                    uiplusone = self.rewards[i]
                else:
                    uiplusone = self.rewards[i] + self.gamma*maxvalueaction

                change = uiplusone - self.V[i]
                changesinu[i] = abs(change)
                self.policy[i] = indexOfMaxValueAction
                self.V[i] = uiplusone
        
            if (max(changesinu) < stoppingconditionvalue):
                stoppingcondition = True
    



    def play_game(self, display=False):
        '''
        Play through one episode of the game under the current policy
        display=True results in displaying the current policy performed on a randomly generated environment in the terminal.
        '''
        self.env.reset()
        episodes = []
        finished = False

        curr_state = self.env.s
        total_reward = 0

        while not finished:
            # display current state
            if display:
                system('cls')
                self.env.render()
                sleep(0.1)

            # find next state
            action = self.policy[curr_state]
            try:
                new_state, reward, finished, info = self.env.step(action)
            except:
                new_state, reward, finished, info, _ = self.env.step(action)
            reward = self.rewards[new_state] # Rewards are realized by entering a new state
            total_reward += reward
            episodes.append([new_state, action, reward])
            curr_state = new_state # Set the current state equal to the new state for the next while loop iteration

        # display end result
        if display:
            system('cls')
            self.env.render()

        print(f"Total Reward from this run: {total_reward}")
        return episodes

    # averages rewards over a number of episodes
    def compute_episode_rewards(self, num_episodes=100, step_limit=1000):
        '''
        Computes the mean, variance, and maximum of episode reward over num_episodes episodes
        '''
        total_rewards = np.zeros(num_episodes);steps_taken = np.zeros(num_episodes)
        for episode in range(num_episodes):
            self.env.reset()
            finished = False
            num_steps = 0
            curr_state = self.env.s
            while not finished and num_steps < step_limit:
                action = self.policy[curr_state]
                try:
                    new_state, reward, finished, info = self.env.step(action)
                except:
                    new_state, reward, finished, info, _ = self.env.step(action)
                reward = self.rewards[new_state]
                total_rewards[episode] += reward
                num_steps += 1
                curr_state = new_state # Set the current state equal to the new state for the next while loop iteration
                

            if reward != 0:
                # If the agent falls into a hole and gets a reward of zero, then record that as zero (already the value of the array)
                # Otherwise, if they do eveutally reach the goal, then record then number of steps taken
                steps_taken[episode] = num_steps

        return np.mean(total_rewards), np.var(total_rewards), np.max(total_rewards), steps_taken


    def print_rewards_info(self, num_episodes=100, step_limit=1000):
        '''
        Prints information from compute_episode_rewards
        '''
        mean, var, best, steps_array = self.compute_episode_rewards(num_episodes=num_episodes, step_limit=step_limit)
        print(f"Mean of Episode Rewards: {mean:.2f}, Variance of Episode Rewards: {var:.2f}, Best Episode Reward: {best}")

# Model free reinforcement learning
class QLearning:
    """
    """

    def __init__(self, env, gamma=0.95, epsilon=0.01):
        """
        Initialize policy, environment, and Q table (Q)
        """
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n 
        self.Q = np.zeros((self.num_states, self.num_actions)) # A 2d-array of states and the values of each action
        self.state_action_counter = np.zeros((self.num_states, self.num_actions))   # keeps track of k_sa
        self.gamma = gamma # Discount rate
        self.epsilon = epsilon # Exploration rate
        self.rewards = {state:0 for state in range(self.num_states)} # The reward function R(s)
        self.terminal_states = {state:False for state in range(self.num_states)} # Returns a True or False indicating if the game state provided 
        # is in a terminal state (i.e. no further actions can be taken)
        
        for state,actions in env.P.items():
            for action,action_data in actions.items():
                prob, state, r, is_end = action_data[0]
                if is_end==True:
                    self.terminal_states[state] = True # Record if terminal state (ice hole or goal)
                    if r == 1:
                        self.rewards[state] = 1 # If a goal state, then R(s) = 1, else R(s) left at 0


    def choose_action(self, state: State) -> Action:
        """
        Returns action based on Q-values using the epsilon-greedy exploration strategy
        """

        argmaxdecision = np.random.choice([1,0], p = [(1-self.epsilon), self.epsilon])

        retAction = 0
        
        if (argmaxdecision):
            maxvalue = max(self.Q[state]) # I used this max function after consulting my friend Sorin 

            maxindices = []
            for action in range(self.num_actions):
                if self.Q[state][action] == maxvalue: 
                    maxindices.append(action)

            retAction = maxindices[np.random.randint(len(maxindices))]
        else: 
            retAction = np.random.randint(self.num_actions)
        
        return retAction


    def q_learning(self, num_episodes=10000, interval=1000, display=False, step_limit=10000):
        """
        Implement the tabular update for the table of Q-values, stored in self.Q

        Boilerplate code of running several episodes and retrieving the (s, a, r, s') transitions has already been done
        for you.
        """
        mean_returns = []
        for e in range(1, num_episodes+1):
            self.env.reset()
            finished = False

            curr_state = self.env.s
            num_steps = 0

            while not finished and num_steps < step_limit:
                # display current state
                if display:
                    system('cls')
                    self.env.render()
                    sleep(1)

                action = self.choose_action(curr_state)
                try:
                    next_state, reward, finished, info = self.env.step(action)
                except:
                    next_state, reward, finished, info, _ = self.env.step(action)
                
                reward = self.rewards[next_state]
                
                self.state_action_counter[curr_state][action] += 1 # Update the state_action_counter
                
                # update Q values. Use the alpha schedule given here. k_SA = how many time we took action A at state S
                alpha = min(0.1, 10 / self.state_action_counter[curr_state][action] ** 0.8)
                
                # Q-learning update rule
                ##########################
                ##### YOUR CODE HERE #####
                ##########################
                
                self.Q[curr_state][action] = (1-alpha)*self.Q[curr_state][action] + alpha*(reward + self.gamma*max(self.Q[next_state])) # I used this max function after consulting with my friend Sorin Choi 
                
                num_steps += 1
                curr_state = next_state

            # run tests every interval episodes
            if e % interval == 0:
                print(str(e)+"/"+str(num_episodes),end=" ")
                mean, var, best = self.compute_episode_rewards(num_episodes=100)
                mean_returns.append(mean)

        return mean_returns

    # averages rewards over a number of episodes
    def compute_episode_rewards(self, num_episodes=100, step_limit=1000):
        '''
        Computes the mean, variance, and maximum of episode reward over num_episodes episodes
        '''
        total_rewards = np.zeros(num_episodes)
        for episode in range(num_episodes):
            self.env.reset()
            finished = False
            num_steps = 0
            curr_state = self.env.s
            while not finished and num_steps < step_limit:
                best_actions = np.argwhere(self.Q[curr_state] == np.amax(self.Q[curr_state])).flatten()
                action = np.random.choice(best_actions)
                try:
                    next_state, reward, finished, info = self.env.step(action)
                except:
                    next_state, reward, finished, info, _ = self.env.step(action)
                reward = self.rewards[next_state]
                total_rewards[episode] += reward
                num_steps += 1
                curr_state = next_state

        mean, var, best = np.mean(total_rewards), np.var(total_rewards), np.max(total_rewards)
        print(f"Mean of Episode Rewards: {mean:.2f}, Variance of Episode Rewards: {var:.2f}, Best Episode Reward: {best}")
        return mean, var, best


if __name__ == "__main__":
    
    ### Part 3 - Value Iteration ###
    #env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True) # Set up the Frozen lake environmnet 
    #env.reset()

    #print("Testing Value Iteration...")
    #sleep(1)
    #my_policy = DynamicProgramming(env, gamma=0.9, epsilon=0.001) # Instantiate class object
    #my_policy.value_iteration() # Iterate to derive the final policy
    #my_policy.play_game() # Play through one episode of the game under the current policy
    #my_policy.print_rewards_info() # Prints information from compute_episode_rewards
    #sleep(1)

    # Compute the episode rewards over 1000 episodes of game playing
    #mean_val, var_val, max_val, num_steps_array =  my_policy.compute_episode_rewards(num_episodes=1000, step_limit=1000)
    #print(f"Mean of Episode Rewards: {mean_val:.2f}, Variance of Episode Rewards: {var_val:.2f}, Best Episode Reward: {max_val}")
    
    import matplotlib.pyplot as plt

    
    
    ### Part 4 - Model Free Q-Learning ###
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    env.reset()

    print("Testing Q-Learning...")
    sleep(1)
    my_policy = QLearning(env, gamma=0.9, epsilon=0.01) # Instanciate a new class object with the Q learning methods
    my_policy.q_learning()
    print(my_policy.Q)





#####################################
# Question 4.1: Logistic Regression #
#####################################

### Package Imports ###
import pandas as pd
from typing import Tuple
### Package Imports ###

if __name__ == "__main__":
    
    ### Package Imports ###
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.linear_model import LogisticRegression
    ### Package Imports ###
    
    ## Starter-code DO NOT EDIT
    iris = datasets.load_iris() # Load the dataset from sklearn
    X = pd.DataFrame(iris.data) # Convery to a pandas dataframe
    X.columns = pd.Series(iris.feature_names).str.replace(" (cm)","",regex=False).str.replace(" ","_") # Add col headers
    X = X.loc[:,["petal_length","petal_width"]] # Subset for only the petal features
    y = iris.target # Extract the response variable
    X = X.loc[y!=0,:];y = y[y!=0] # Remove one of the classes, the one labeled 0 for this data analysis
    y = y-1 # Adjust labels to be 0 and 1 instead of 1 and 2 in the response variable




######################################
# Question 4.2: Perceptron Algorithm #
######################################

# Toy data set for practice with the perceptron algorithm
x1 = [1,1,2,2,2,3,3] # First data set feature
x2 = [0,2,1,2,3,3,1] # Second data set feature
x0 = [1]*len(x1) # Create a vector of 1s to add a bias term
X = pd.DataFrame({"x0":x0, "x1":x1,"x2":x2}) # Create a pandas DF for the x with the bias term included
y = np.array([ 1,  1, -1, -1,  1, -1, -1]) # Response vector
weight_vector = np.array([1,1,0]) # In the form of (x0, x1, x2)



def perceptron_iter(X:pd.Series, y:float, weight_vector:np.array)->Tuple[np.array,bool]:
    """Takes in an X pd.Series representing the feature vector for 1 obs, a y value of that obs, and a weight vector. Performs 
    1 iteration of the perceptron algorithm to update the weight vector using this data point from the broader dataset"""
    change=False # Also return a boolean value indicating if the weight vector has been changed from input to output

    ##########################
    ##### YOUR CODE HERE #####
    ##########################
    
    X = X.astype(str).astype(float)
    featurelist = np.array(X)
    dotproduct = np.dot(featurelist, weight_vector)
    if (dotproduct>=0):
        if (y==1):
            return weight_vector, change
    if (dotproduct<0):
        if (y==-1):
            return weight_vector, change
    
    update = y*featurelist
    updated_weight_vector = np.add(weight_vector, update)
    change=True
    
    
    return updated_weight_vector, change

def run_perceptron_algo(X:pd.DataFrame, y:np.array, weight_vector:np.array, max_iter:int=1000)->Tuple[np.array,int]:
    """Takes in a dataset denoted by X and y, with a starting weight vector and runs the perceptron algorithm until convergence
    or a max iteration threshold has been exceeded, returns the fitted weight vector and the number of iterations required"""
    iterations=0
    
    # Hint: Call the perceptron_iter() helper function from above
    
    while (iterations < max_iter):
        convergence = True
        iterations += 1

        numberOfObjects = len(X)
        for row in range(numberOfObjects):
            updated_weight_vector, change = perceptron_iter(X.loc[row], y[row], weight_vector) 
            weight_vector = updated_weight_vector 
            if (change == True):
                convergence = False

        if (convergence): 
            return weight_vector, iterations

    return weight_vector, iterations

if __name__ == "__main__":
    # Run the perceptron algorithm and print the results
    weight_vector, iterations = run_perceptron_algo(X, y, weight_vector)
    print("Iterations Until Convergence: "+str(iterations))
    print("Classifier Weights:\n"+str(weight_vector))
    
    # Add another observation and re-run
    X.loc[7,:]=[1,2.5,0] # Append a new row to the dataframe
    y=np.append(y,1) # Append a new response variable observation to the y-vector
    
    # Run the perceptron algorithm and print the results
    weight_vector, iterations = run_perceptron_algo(X, y, weight_vector)
    print("Iterations Until Convergence: "+str(iterations))
    print("Classifier Weights:\n"+str(weight_vector))
    
    # Plot the set of data points including the newly added point at (2.5, 1)
    plt.figure(figsize=(8,6)) # Configure plot size and add in background classification region shading
    colors = ['red' if term==-1 else 'blue' for term in y ] # Determine colors for data points based on y
    plt.scatter(X.iloc[:,1],X.iloc[:,2],color=colors) # Plot the originial data points
    plt.xlabel("X1");plt.ylabel("X2");plt.title("X1, X2 Feature Space with Added Point")
    plt.show()           


def fetch_np_array(url: str, path: str) -> np.ndarray:
    """
    This function retrieves the data from the given url and caches the data locally in the path so that we do not
    need to repeatedly download the data every time we call this function.

    Args:
        url: link from which to retrieve data
        path: path on local desktop to save file

    Returns:
        Numpy array that is fetched from the given url or retrieved from the cache.
    """
    fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


class MNISTClassificationModel(abc.ABC):
    neural_net_model = None

    @abc.abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Args:
            X_train: Numpy array of shape (num_samples, 28, 28) of integer values, from 0...255 for color intensity.
            y_train: Numpy array of shape (num_samples,), containing integer labels
        """
    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: Numpy array of shape (num_samples, 28, 28) of integer values, from 0...255 for color intensity.

        Returns:
            y: Numpy array of shape (num_samples,) of integer-valued labels.
        """

class TensorFlowMNISTClassifier(MNISTClassificationModel):
    def __init__(self):
        """
        Initialize your `neural_net_model`.
        """

        self.neural_net_model = tf.keras.Sequential()
        #self.neural_net_model.add(tf.keras.layers.Reshape(target_shape=(32 * 32,), input_shape=(28, 28)))
        self.neural_net_model.add(tf.keras.layers.Dense(units=256, activation='relu'))   # credit here given to https://www.projectpro.io/recipes/build-simple-neural-network-tensorflow
        #self.neural_net_model.add(tf.keras.layers.Dense(units=192, activation='relu'))
        #self.neural_net_model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        #self.neural_net_model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

        
        #self.neural_net_model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
        self.neural_net_model.add(tf.keras.layers.Flatten())
        self.neural_net_model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        self.neural_net_model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
        self.neural_net_model.add(tf.keras.layers.Dense(units=1, activation='relu'))

        

        assert isinstance(self.neural_net_model, tf.Module)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.neural_net_model.compile(optimizer='adam', 
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        history = self.neural_net_model.fit(X_train, y_train, epochs=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Network forward pass, for an input dataset X, return the output predictions"""
        y_hat = self.neural_net_model.predict(X) # credit goes to https://www.youtube.com/watch?v=6_2hzRopPbQ
        returny_hat = (np.rint(y_hat)).astype(int)
        return returny_hat


def train_and_evaluate_mnist_model(
        model: MNISTClassificationModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
        ) -> float:
    """
    Evaluates the model provided to this function given the appropriate inputs.

    Returns:
        Prediction accuracy on test set
    """
    start = time.time()
    model.train(X_train, y_train)
    end = time.time()
    print(f"Training time is {end - start} seconds.")
    y_pred = model.predict(X_test)
    number_correct = (y_pred == y_test).sum()
    return number_correct / len(y_test)

if __name__ == "__main__":
    # Obtain data from the web
    pset_5_cache = ""
    X_train = fetch_np_array(
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", pset_5_cache
    )[0x10:].reshape((-1, 28, 28))
    y_train = fetch_np_array("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", pset_5_cache)[8:]
    X_test = fetch_np_array(
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", pset_5_cache
    )[0x10:].reshape((-1, 28, 28))
    y_test = fetch_np_array("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", pset_5_cache)[8:]

    # Initialize tensorflow model for training and evaluation
    tf_model = TensorFlowMNISTClassifier()
    tf_accuracy = train_and_evaluate_mnist_model(tf_model, X_train, y_train, X_test, y_test)
    print(f"Accuracy of tf model is {tf_accuracy}.")

