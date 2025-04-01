import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class TrafficLightOptimizer:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Input for current traffic state
        traffic_input = Input(shape=(self.state_size,))
        
        # Input for previous action
        prev_action_input = Input(shape=(self.action_size,))
        
        # Input for time of day
        time_input = Input(shape=(1,))
        
        # Combine all inputs
        merged = Concatenate()([traffic_input, prev_action_input, time_input])
        
        # Neural network layers
        x = Dense(128, activation='relu')(merged)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        output = Dense(self.action_size, activation='softmax')(x)
        
        model = Model(inputs=[traffic_input, prev_action_input, time_input], outputs=output)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        traffic_state, prev_action, time = state
        act_values = self.model.predict([
            np.array([traffic_state]),
            np.array([prev_action]),
            np.array([time])
        ])
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            traffic_state, prev_action, time = state
            next_traffic_state, next_prev_action, next_time = next_state
            
            target = reward
            if not done:
                target = (reward + self.gamma *
                         np.amax(self.target_model.predict([
                             np.array([next_traffic_state]),
                             np.array([next_prev_action]),
                             np.array([next_time])
                         ])[0]))
            
            target_f = self.model.predict([
                np.array([traffic_state]),
                np.array([prev_action]),
                np.array([time])
            ])
            target_f[0][action] = target
            
            self.model.fit([
                np.array([traffic_state]),
                np.array([prev_action]),
                np.array([time])
            ], target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

