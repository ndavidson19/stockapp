# The backend for my stocktrading application

The goal is to have a machine learning agent be able to made live trades.
These trades will be viewable on the frontend, which will be a React app.


The backend will have a ensemble model, made up of 5 different models ARIMA, XGBOOST, LSTM, Fourier Transforms, and SVM with a Polynomial Kernel. Each of these has its own advantages when it comes to stocktrading

The ensemble model is used to make a prediction on the next day's stock price. This prediction is used to help inform the reinforcement agents decision on whether to buy, sell, or hold a stock. The ensemble model is used to evaluate the performance of the learned policy by using it to make predictions about the future state of the system based on the current state and the actions taken by the RL algorithm. You can then compare the predictions made by the ensemble model with the actual outcomes to see how well the policy is performing. This is called model-free reinforcement learning. The ensemble model will also have a sentiment component scores the quality of the news articles. 

A neural network will be incorporated into the RL agent, as it can be used as the function approximator which represents the policy or value function. For example, you could use a neural network to predict the action to take at each step based on the current state of the system, or to estimate the expected long-term reward for each action in each state.

Once the agent has learned a policy, it can be used to make predictions about the future state of the system based on the current state and the actions taken by the RL algorithm. You can then compare the predictions made by the RL agent with the actual outcomes to see how well the policy is performing. This is called model-based reinforcement learning. Since the results are continuously updated live with the streaming of trading data, the agent can learn from its mistakes and improve its performance over time.

The model will be deployed on the cloud and be able to make inferences on the fly. 



