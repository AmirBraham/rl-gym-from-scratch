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
    
if __name__ == "__main__":
    env = Environment(10)
    state = env.reset()