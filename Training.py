from Environment import Environment
from Agent import Agent
from ExperienceReplay import ExperienceReplay

if __name__ == "__main__":
    gird_size = 5
    environment = Environment(gird_size,render_on=True)
    agent = Agent(gird_size,epsilon=.9,epsilon_decay=0.998,epsilon_end=0.01)
    experienceReplay = ExperienceReplay(capacity=10000,batch_size=32)
    episodes = 5000
    max_steps = 200
    for ep in range(episodes):
        state = environment.reset()
        for step in range(max_steps):
            print('Episode:', ep)
            print('Step:', step)
            print('Epsilon:', agent.epsilon)
            action = agent.get_action(state)
            reward,next_state,done = environment.step(action)
            experienceReplay.add_experience(state,action,reward,next_state,done)
            if(experienceReplay.can_provide_sample()):
                experiences = experienceReplay.sample_batch()
                agent.learn(experiences)
            state = next_state
            if done:
                break
            
        agent.save(f'models/model_{gird_size}.h5')