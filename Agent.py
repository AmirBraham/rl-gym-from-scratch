from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential,load_model
import numpy as np 
class Agent:
    def __init__(self,grid_size,epsilon=1,epsilon_decay=0.998,epsilon_end=0.1,gamma=0.99) -> None:
        self.grid_size = grid_size
        self.model = self.build_model()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.gamma = gamma
        
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
            action = np.random.randint(0,4)
            return action
        
        state = np.expand_dims(state,axis=0)
        q_values = self.model.predict(state,verbose=0)
        action = np.argmax(q_values[0])
        #Indices 0, 1, 2, and 3 will be mapped to up, down, left, and right respectively.
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        return action
    
    def learn(self,experiences):
        states = np.array([experience.state for experience in experiences])
        actions = np.array([experience.action for experience in experiences])
        rewards = np.array([experience.reward for experience in experiences])
        next_states = np.array([experience.next_state for experience in experiences])
        dones = np.array([experience.done for experience in experiences])
        current_q_values = self.model.predict(states,verbose=0)
        next_q_values = self.model.predict(next_states,verbose=0)
        target_q_values = current_q_values.copy()
        for i in range(len(experiences)):
            if dones[i]:
                # If the episode is done, there is no next Q-value
                # [i, actions[i]] is the numpy equivalent of [i][actions[i]]
                target_q_values[i, actions[i]] = rewards[i]
            else:
                # The updated Q-value is the reward plus the discounted max Q-value for the next state
                # [i, actions[i]] is the numpy equivalent of [i][actions[i]]
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        self.model.fit(states,target_q_values,epochs=1,verbose=0)
    def load(self,file_path):
        self.model = load_model(file_path)
        
    def save(self,file_path):
        self.model.save(file_path)