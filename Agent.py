from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np 
class Agent:
    def __init__(self,grid_size,epsilon=1,epsilon_decay=0.998,epsilon_end=0.1) -> None:
        self.grid_size = grid_size
        self.model = self.build_model()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        
    def build_model(self):
        # Since our agent has 4 possible moves (up,right,down,left) , the final layer of our NN should have 4 activation neurons
        model = Sequential([
            Dense(128,input_shape=(self.grid_size**2,),activation='relu'),
            Dense(64,activation='relu'),
            Dense(4,activation='linear')
        ])
        
        model.compile(optimizer='adam',loss='mse')
        return model
    
    def get_action(self,state):
        if np.random.rand() < self.epsilon:
            action = np.random.randin(0,4)
            return action
        
        state = np.expand_dims(state,axis=0)
        q_values = self.model.predict(state,verbose=0)
        action = np.argmax(q_values[0])
        #Indices 0, 1, 2, and 3 will be mapped to up, down, left, and right respectively.
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        return action