#module for optimization
from bayes_opt import BayesianOptimization, UtilityFunction
# module for logging data 
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
# module for retriving datat 
from bayes_opt.util import load_logs

# import the function to be optimized
from main3 import *

# logger listes to Events.OPTIMIZATION_STEP
# logger will not look back at previous probede points `

# # ------------------- MLP -------------------
# pbounds = {"batch_size" : (32, 128), "learning_rate" : (0.001, 0.009), "num_epochs": (1, 5)}
# def train_rnn_wrapper(batch_size, learning_rate, num_epochs):
#     nn_type = "mlp" # elman_scratch, elman_torch, lstm, mlp
#     batch_size = int(round(batch_size))
#     num_epochs = int(round(num_epochs))
#     epoch_num, train_loss, val_accuracy = train_rnn(batch_size, learning_rate, num_epochs, nn_type)
#     return val_accuracy


# # create instance of optimizer 
# optimizer1 = BayesianOptimization(
#     f = train_rnn_wrapper,
#     pbounds = pbounds,
#     random_state = 1
# )

# # create UtilityFunction object for aqu. function
# utility = UtilityFunction(kind = "ei", xi= 0.02)

# # set gaussian process parameter
# optimizer1.set_gp_params(alpha = 1e-6)

# # create logger 
# logger = JSONLogger(path = "./tunning_mlp.log")
# optimizer1.subscribe(Events.OPTIMIZATION_STEP, logger)

# # initial search 
# optimizer1.maximize(
#     init_points = 5, # number of random explorations before bayes_opt
#     n_iter = 15, # number of bayes_opt iterations
# )

# # print out the data from the initial run to check if bounds need update 
# for i, param in enumerate(optimizer1.res):
#     print(f"Iteration {i}: \n\t {param}")

# # get best parameter
# print("Best Parameters found: ")
# print(optimizer1.max)


# # ------------------- LSTM -------------------
# # bounded parameter regions 
# pbounds = {"batch_size" : (32, 128), "learning_rate" : (0.001, 0.009), "num_epochs": (1, 5)}
# # define wrapped funciton
# def train_lstm_rnn_wrapper(batch_size, learning_rate, num_epochs):
#     nn_type = "lstm" # elman_scratch, elman_torch, lstm, mlp
#     batch_size = int(round(batch_size))
#     num_epochs = int(round(num_epochs))
#     epoch_num, train_loss, val_accuracy = train_rnn(batch_size, learning_rate, num_epochs, nn_type)
#     return val_accuracy

# # create instance of optimizer 
# optimizer2 = BayesianOptimization(
#     f = train_lstm_rnn_wrapper,
#     pbounds = pbounds,
#     random_state = 1
# )

# # create UtilityFunction object for aqu. function
# utility = UtilityFunction(kind = "ei", xi= 0.02)

# # set gaussian process parameter
# optimizer2.set_gp_params(alpha = 1e-6)

# # create logger 
# logger = JSONLogger(path = "./tunning_lstm.log")
# optimizer2.subscribe(Events.OPTIMIZATION_STEP, logger)

# # initial search 
# optimizer2.maximize(
#     init_points = 5, # number of random explorations before bayes_opt
#     n_iter = 15, # number of bayes_opt iterations
# )

# # print out the data from the initial run to check if bounds need update 
# for i, param in enumerate(optimizer2.res):
#     print(f"Iteration {i}: \n\t {param}")

# # get best parameter
# print("Best Parameters found: ")
# print(optimizer2.max)



# ------------------- Elman torch -------------------
pbounds = {"batch_size" : (32, 128), "learning_rate" : (0.001, 0.009), "num_epochs": (1, 5)}
def train_elman_rnn_wrapper(batch_size, learning_rate, num_epochs):
    nn_type = "elman_torch" # elman_scratch, elman_torch, lstm, mlp
    batch_size = int(round(batch_size))
    num_epochs = int(round(num_epochs))
    epoch_num, train_loss, val_accuracy = train_rnn(batch_size, learning_rate, num_epochs, nn_type)
    return val_accuracy


# create instance of optimizer 
optimizer3 = BayesianOptimization(
    f = train_elman_rnn_wrapper,
    pbounds = pbounds,
    random_state = 1
)

# create UtilityFunction object for aqu. function
utility = UtilityFunction(kind = "ei", xi= 0.02)

# set gaussian process parameter
optimizer3.set_gp_params(alpha = 1e-6)

# create logger 
logger = JSONLogger(path = "./tunning_elman.log")
optimizer3.subscribe(Events.OPTIMIZATION_STEP, logger)

# initial search 
optimizer3.maximize(
    init_points = 5, # number of random explorations before bayes_opt
    n_iter = 15, # number of bayes_opt iterations
)

# print out the data from the initial run to check if bounds need update 
for i, param in enumerate(optimizer3.res):
    print(f"Iteration {i}: \n\t {param}")

# get best parameter
print("Best Parameters found: ")
print(optimizer3.max)


# # ------------------- Example 1 -------------------
# # bounded parameter regions 
# pbounds = {"pop_size" : (50, 250), "m_rate" : (0.01, 0.5), "p_gaussian": (0.01, 0.5)}

# # define wrapped funciton
# def evo_strategies_wrapper(pop_size, m_rate,  p_gaussian):
#     mutation_operator = adaptive_ensemble_mutation
#     survival_selection = two_archives_survival
#     pop_size = int(round(pop_size))
#     max_gen = 50
#     enemies = [4, 5, 6, 7]
#     ind,fitness=evo_strategies(pop_size, m_rate, max_gen, mutation_operator=mutation_operator,
#                          survival_selection=survival_selection, enemies=enemies, fitness_threshold=60,local=False)
#     return fitness [-1]

# # create instance of optimizer 
# optimizer1 = BayesianOptimization(
#     f = evo_strategies_wrapper,
#     pbounds = pbounds,
#     random_state = 1
# )

# # create UtilityFunction object for aqu. function
# utility = UtilityFunction(kind = "ei", xi= 0.02)

# # set gaussian process parameter
# optimizer1.set_gp_params(alpha = 1e-6)

# # create logger 
# logger = JSONLogger(path = "./tunning1.log")
# optimizer1.subscribe(Events.OPTIMIZATION_STEP, logger)

# # initial search 
# optimizer1.maximize(
#     init_points = 5, # number of random explorations before bayes_opt
#     n_iter = 15, # number of bayes_opt iterations
# )

# # print out the data from the initial run to check if bounds need update 
# for i, param in enumerate(optimizer1.res):
#     print(f"Iteration {i}: \n\t {param}")

# # get best parameter
# print("Best Parameters found: ")
# print(optimizer1.max)



# # ------------------- Example 2 -------------------

# # bounded parameter regions 
# pbounds = {"max_gen": (50, 150) , "m_rate" : (0.01, 0.5)}
# enemies = [4, 5, 6, 7]
# # define wrapped funciton
# def evo_strategies_wrapper(max_gen, m_rate):
#     pop_size = 50 
#     max_gen = int(round(max_gen))
#     mutation_operator = swag_mutation
#     survival_selection = two_archives_survival
#     enemies = [4, 5, 6, 7]
#     ind,fitness=evo_strategies(pop_size, m_rate, max_gen, mutation_operator=mutation_operator,
#                          survival_selection=survival_selection, enemies=enemies, fitness_threshold=60,local=False)
#     return fitness [-1]

# # create instance of optimizer 
# optimizer2 = BayesianOptimization(
#     f = evo_strategies_wrapper,
#     pbounds = pbounds,
#     random_state = 1
# )

# # create UtilityFunction object for aqu. function
# utility = UtilityFunction(kind = "ei", xi= 0.02)

# # set gaussian process parameter
# optimizer2.set_gp_params(alpha = 1e-6)

# # create logger 
# logger = JSONLogger(path = "./tunning2.log")
# optimizer2.subscribe(Events.OPTIMIZATION_STEP, logger)

# # initial search 
# optimizer2.maximize(
#     init_points = 5, # number of random explorations before bayes_opt
#     n_iter = 15, # number of bayes_opt iterations
# )

# # print out the data from the initial run to check if bounds need update 
# for i, param in enumerate(optimizer2.res):
#     print(f"Iteration {i}: \n\t {param}")

# # get best parameter
# print("Best Parameters found: ")
# print(optimizer2.max)




# # ------------------- Example 3 -------------------
# # bounded parameter regions 
# pbounds = { "m_rate" : (0.01, 0.5)}

# # define wrapped funciton
# def evo_strategies_wrapper(m_rate):
#     pop_size = 50 
#     max_gen = 50
#     mutation_operator = multi_adaptive_mutation
#     survival_selection = two_archives_survival
#     enemies = [4, 5, 6, 7]
#     ind,fitness=evo_strategies(pop_size, m_rate, max_gen, mutation_operator=mutation_operator,
#                          survival_selection=survival_selection, enemies=enemies, fitness_threshold=60,local=False)
#     return fitness [-1]

# # create instance of optimizer 
# optimizer3 = BayesianOptimization(
#     f = evo_strategies_wrapper,
#     pbounds = pbounds,
#     random_state = 1
# )

# # create UtilityFunction object for aqu. function
# utility = UtilityFunction(kind = "ei", xi= 0.02)

# # set gaussian process parameter
# optimizer3.set_gp_params(alpha = 1e-6)

# # create logger 
# logger = JSONLogger(path = "./tunning3.log")
# optimizer3.subscribe(Events.OPTIMIZATION_STEP, logger)

# # initial search 
# optimizer3.maximize(
#     init_points = 5, # number of random explorations before bayes_opt
#     n_iter = 15, # number of bayes_opt iterations
# )

# # print out the data from the initial run to check if bounds need update 
# for i, param in enumerate(optimizer3.res):
#     print(f"Iteration {i}: \n\t {param}")

# # get best parameter
# print("Best Parameters found: ")
# print(optimizer3.max)






# # ------------------- Example 4 -------------------
# # bounded parameter regions 
# pbounds = { "m_rate" : (0.01, 50),"generation_threshold" : (0.5, 0.7), "scale": (0.1, 0.5), "polynomial_eta" : (10, 30)}

# # define wrapped funciton
# def evo_strategies_wrapper(m_rate, generation_threshold, scale, polynomial_eta):
#     pop_size = 50 
#     max_gen = 50
#     mutation_operator = cauchy_polynomial_mutation
#     survival_selection = two_archives_survival
#     polynomial_eta = int(round(polynomial_eta))
#     enemies = [4, 5, 6, 7]
#     return evo_strategies(pop_size, m_rate, max_gen, mutation_operator=mutation_operator,
#                          survival_selection=survival_selection, enemies=enemies, local=False,
#                          generation_threshold=generation_threshold, scale=scale,
#                          polynomial_eta=polynomial_eta)

# # create instance of optimizer 
# optimizer4 = BayesianOptimization(
#     f = evo_strategies_wrapper,
#     pbounds = pbounds,
#     random_state = 1
# )

# # create UtilityFunction object for aqu. function
# utility = UtilityFunction(kind = "ei", xi= 0.02)

# # set gaussian process parameter
# optimizer3.set_gp_params(alpha = 1e-6)

# # create logger 
# logger = JSONLogger(path = "./tunning4.log")
# optimizer4.subscribe(Events.OPTIMIZATION_STEP, logger)

# # initial search 
# optimizer4.maximize(
#     init_points = 5, # number of random explorations before bayes_opt
#     n_iter = 15, # number of bayes_opt iterations
# )

# # print out the data from the initial run to check if bounds need update 
# for i, param in enumerate(optimizer4.res):
#     print(f"Iteration {i}: \n\t {param}")

# # get best parameter
# print("Best Parameters found: ")
# print(optimizer4.max)