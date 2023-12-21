import numpy as np 

class Environment:
    def __init__(self,grid_size,render_on=False) -> None:
        self.grid_size = grid_size
        self.grid = []
        self.render_on = render_on
        self.agent_location = None
        self.goal_location = None
        
    def reset(self) -> [int]:
        self.grid = np.zeros((self.grid_size,self.grid_size))
        self.agent_location = self.add_agent()
        self.goal_location = self.add_goal()
        if(self.render_on):
            print(self)
        return self.get_state()
        
    def add_agent(self) -> None:
        # The player will be represented by 1 in the grid 
        location = (np.random.randint(0,self.grid_size),np.random.randint(0,self.grid_size))
        self.grid[location[0]][location[1]] = 1
        return location
    
    def add_goal(self):
        # The goal will be represented by a -1 in the grid
        location = (np.random.randint(0,self.grid_size),np.random.randint(0,self.grid_size))
        while self.grid[location[0]][location[1]] == 1:
            location = (np.random.randint(0,self.grid_size),np.random.randint(0,self.grid_size))
        self.grid[location[0]][location[1]] = -1
        return location
    
    def __str__(self) -> str:
        grid = self.grid.astype(int).tolist()
        representation = ""
        for row in grid:
            representation += ' '.join([str(r) for r in row]) + "\n"
        return representation+"\n"
    
    def get_state(self):
        # Neural networks usually require 1d array
        return self.grid.flatten()
    
    def move_agent(self, action):
        hasReachedGoal = False
        reward = 0
        # Map agent action to the correct movement
        moves = {
            0: (-1, 0), # Up
            1: (1, 0),  # Down
            2: (0, -1), # Left
            3: (0, 1)   # Right
        }
        
        previous_location = self.agent_location
        
        # Determine the new location after applying the action
        move = moves[action]
        new_location = (previous_location[0] + move[0], previous_location[1] + move[1])
        
        # Check for a valid move
        if self.is_valid_location(new_location):
            # Remove agent from old location
            self.grid[previous_location[0]][previous_location[1]] = 0
            
            # Add agent to new location
            self.grid[new_location[0]][new_location[1]] = 1
            
            # Update agent's location
            self.agent_location = new_location
            if self.agent_location == self.goal_location:
                hasReachedGoal = True
                reward = 100
            else:
                newDistance = np.abs(self.goal_location[0] - self.agent_location[0]) + np.abs(self.goal_location[1] - self.agent_location[1])
                prevDistance = np.abs(self.goal_location[0] - previous_location[0]) + np.abs(self.goal_location[1] - previous_location[1])
                reward = (prevDistance - newDistance) - 0.1 # the .1 serves to avoid infinite loops
        else:
            reward = -3
        return reward, hasReachedGoal
            
    def is_valid_location(self, location):
        # Check if the location is within the boundaries of the grid
        if (0 <= location[0] < self.grid_size) and (0 <= location[1] < self.grid_size):
            return True
        else:
            return False
        
    def step(self,action):
        reward , hasReachedGoal = self.move_agent(action)
        next_state = self.get_state()
        if self.render_on:
            print(self)
        return reward,next_state,hasReachedGoal 
    
if __name__ == "__main__":
    env = Environment(10)
    state = env.reset()