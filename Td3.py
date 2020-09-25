import tensorflow as tf
import numpy as np
import gym
from tensorflow.keras.models import load_model
#!pip3 install box2d-py

print(tf.config.list_physical_devices('GPU'))

env= gym.make("LunarLanderContinuous-v2")
state_low = env.observation_space.low
state_high = env.observation_space.high
action_low = env.action_space.low 
action_high = env.action_space.high
print(state_low)
print(state_high)
print(action_low)
print(action_high)


class RBuffer():
  def __init__(self, maxsize, statedim, naction):
    self.cnt = 0
    self.maxsize = maxsize
    self.state_memory = np.zeros((maxsize, *statedim), dtype=np.float32)
    self.action_memory = np.zeros((maxsize, naction), dtype=np.float32)
    self.reward_memory = np.zeros((maxsize,), dtype=np.float32)
    self.next_state_memory = np.zeros((maxsize, *statedim), dtype=np.float32)
    self.done_memory = np.zeros((maxsize,), dtype= np.bool)

  def storexp(self, state, next_state, action, done, reward):
    index = self.cnt % self.maxsize
    self.state_memory[index] = state
    self.action_memory[index] = action
    self.reward_memory[index] = reward
    self.next_state_memory[index] = next_state
    self.done_memory[index] = 1- int(done)
    self.cnt += 1

  def sample(self, batch_size):
    max_mem = min(self.cnt, self.maxsize)
    batch = np.random.choice(max_mem, batch_size, replace= False)  
    states = self.state_memory[batch]
    next_states = self.next_state_memory[batch]
    rewards = self.reward_memory[batch]
    actions = self.action_memory[batch]
    dones = self.done_memory[batch]
    return states, next_states, rewards, actions, dones


class Critic(tf.keras.Model):
  def __init__(self):
    super(Critic, self).__init__()
    self.f1 = tf.keras.layers.Dense(512, activation='relu')
    self.f2 = tf.keras.layers.Dense(512, activation='relu')
    self.v =  tf.keras.layers.Dense(1, activation=None)

  def call(self, inputstate, action):
    x = self.f1(tf.concat([inputstate, action], axis=1))
    x = self.f2(x)
    x = self.v(x)
    return x


class Actor(tf.keras.Model):
  def __init__(self, no_action):
    super(Actor, self).__init__()    
    self.f1 = tf.keras.layers.Dense(512, activation='relu')
    self.f2 = tf.keras.layers.Dense(512, activation='relu')
    self.mu =  tf.keras.layers.Dense(no_action, activation='tanh')

  def call(self, state):
    x = self.f1(state)
    x = self.f2(x)
    x = self.mu(x)  
    return x

 

class Agent():
  def __init__(self, n_action= len(env.action_space.high)):
    self.actor_main = Actor(n_action)
    self.actor_target = Actor(n_action)
    self.critic_main = Critic()
    self.critic_main2 = Critic()
    self.critic_target = Critic()
    self.critic_target2 = Critic()
    self.batch_size = 64
    self.n_actions = len(env.action_space.high)
    self.a_opt = tf.keras.optimizers.Adam(0.001)
    # self.actor_target = tf.keras.optimizers.Adam(.001)
    self.c_opt1 = tf.keras.optimizers.Adam(0.002)
    self.c_opt2 = tf.keras.optimizers.Adam(0.002)
    # self.critic_target = tf.keras.optimizers.Adam(.002)
    self.memory = RBuffer(1_00_000, env.observation_space.shape, len(env.action_space.high))
    self.trainstep = 0
    #self.replace = 5
    self.gamma = 0.99
    self.min_action = env.action_space.low[0]
    self.max_action = env.action_space.high[0]
    self.actor_update_steps = 2
    self.warmup = 200
    

  def act(self, state, evaluate=False):
      if self.trainstep > self.warmup:
            evaluate = True
      state = tf.convert_to_tensor([state], dtype=tf.float32)
      actions = self.actor_main(state)
      if not evaluate:
          actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=0.1)

      actions = self.max_action * (tf.clip_by_value(actions, self.min_action, self.max_action))
      #print(actions)
      return actions[0]


  def savexp(self,state, next_state, action, done, reward):
        self.memory.storexp(state, next_state, action, done, reward)

  def update_target(self):
    self.actor_target.set_weights(self.actor_main.get_weights())
    self.critic_target.set_weights(self.critic_main.get_weights())
    self.critic_target2.set_weights(self.critic_main2.get_weights())

  
  def train(self):
      if self.memory.cnt < self.batch_size:
        return 


      states, next_states, rewards, actions, dones = self.memory.sample(self.batch_size)
  
      states = tf.convert_to_tensor(states, dtype= tf.float32)
      next_states = tf.convert_to_tensor(next_states, dtype= tf.float32)
      rewards = tf.convert_to_tensor(rewards, dtype= tf.float32)
      actions = tf.convert_to_tensor(actions, dtype= tf.float32)
      #dones = tf.convert_to_tensor(dones, dtype= tf.bool)

      with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            
          target_actions = self.actor_target(next_states)
          target_actions += tf.clip_by_value(tf.random.normal(shape=[*np.shape(target_actions)], mean=0.0, stddev=0.2), -0.5, 0.5)
          target_actions = self.max_action * (tf.clip_by_value(target_actions, self.min_action, self.max_action))
          
          
          target_next_state_values = tf.squeeze(self.critic_target(next_states, target_actions), 1)
          target_next_state_values2 = tf.squeeze(self.critic_target2(next_states, target_actions), 1)
          
          critic_value = tf.squeeze(self.critic_main(states, actions), 1)
          critic_value2 = tf.squeeze(self.critic_main2(states, actions), 1)
          
          next_state_target_value = tf.math.minimum(target_next_state_values, target_next_state_values2)
          
          target_values = rewards + self.gamma * next_state_target_value * dones
          critic_loss1 = tf.keras.losses.MSE(target_values, critic_value)
          critic_loss2 = tf.keras.losses.MSE(target_values, critic_value2)
          


      
      grads1 = tape1.gradient(critic_loss1, self.critic_main.trainable_variables)
      grads2 = tape2.gradient(critic_loss2, self.critic_main2.trainable_variables)
      
      self.c_opt1.apply_gradients(zip(grads1, self.critic_main.trainable_variables))
      self.c_opt2.apply_gradients(zip(grads2, self.critic_main2.trainable_variables))
      
      
      self.trainstep +=1
      
      if self.trainstep % self.actor_update_steps == 0:
                
          with tf.GradientTape() as tape3:
            
              new_policy_actions = self.actor_main(states)
              actor_loss = -self.critic_main(states, new_policy_actions)
              actor_loss = tf.math.reduce_mean(actor_loss)
          
          grads3 = tape3.gradient(actor_loss, self.actor_main.trainable_variables)
          self.a_opt.apply_gradients(zip(grads3, self.actor_main.trainable_variables))

      #if self.trainstep % self.replace == 0:
      self.update_target()
           
      
 


with tf.device('GPU:0'):
    agent = Agent(2)
    tf.random.set_seed(336699)

    episods = 20000
    ep_reward = []
    total_avgr = []
    target = False

    for s in range(episods):
      if target == True:
        break
      total_reward = 0 
      state = env.reset()
      done = False

      while not done:
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.savexp(state, next_state, action, done, reward)
        agent.train()
        state = next_state
        total_reward += reward
        if done:
            ep_reward.append(total_reward)
            avg_reward = np.mean(ep_reward[-100:])
            total_avgr.append(avg_reward)
            print("total reward after {} steps is {} and avg reward is {}".format(s, total_reward, avg_reward))
            if avg_reward == 200:
              target = True



