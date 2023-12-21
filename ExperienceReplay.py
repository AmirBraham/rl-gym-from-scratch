from collections import namedtuple , deque
import random
class ExperienceReplay:
    def __init__(self,capacity,batch_size) -> None:
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.Experience = namedtuple("Experience",["state","action","reward","next_state","done"])
        
    def add_experience(self,state,action,reward,next_state,done):
        experience = self.Experience(state,action,reward,next_state,done)
        self.memory.append(experience)
        
    def sample_batch(self):
        batch = random.sample(self.memory,self.batch_size)
        return batch
    def can_provide_sample(self):
        # Determines if the length of memory has exceeded batch_size
        return len(self.memory) >= self.batch_size