import numpy as np
from smac.model.gaussian_process.kernels import MaternKernel, ConstantKernel, RBFKernel
# from smac.model.gaussian_process.kernels import RBFKernel
from smac.model.gaussian_process.gaussian_process import GaussianProcess
from smac.acquisition.function.expected_improvement import EI
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter, Constant
from ConfigSpace import Configuration, ConfigurationSpace, Float
from matplotlib import pyplot as plt
from smac.intensifier.hyperband import Hyperband, SuccessiveHalving
from smac.model.random_forest import RandomForest
import argparse
from functools import partial
from train_gpt2_class import Trainer, setup_logger, print_with_tabs
from smac import MultiFidelityFacade, RunHistory, Scenario
# from smac.intensifier.hyperband_utils import get_n_trials_for_hyperband_multifidelity
from smac.multi_objective.parego import ParEGO
import logging
from smac.facade.abstract_facade import AbstractFacade
import submitit
import pickle
import torch
from smac.runhistory.dataclasses import TrialValue
import math
from types import SimpleNamespace

global log_file_name

def set_queue(q_, log_folder, maximum_runtime=None):
    global ex
    global q
    if q_ == 'all':
        q = 'alldlc_gpu-rtx2080'
    if q_ == 'ml':
        q = 'mldlc_gpu-rtx2080'
    if q_ == 'mlhiwi':
        q = "mlhiwidlc_gpu-rtx2080"

    if maximum_runtime is None:
        if q == 'alldlc_gpu-rtx2080' or q == 'mlhiwidlc_gpu-rtx2080':
            maximum_runtime = 24*60*1-1
        else:
            maximum_runtime = 24*60*4-1

    ex = submitit.AutoExecutor(folder=log_folder)
    ex.update_parameters(timeout_min=maximum_runtime,
                        slurm_partition=q, #  mldlc_gpu-rtx2080
                        slurm_signal_delay_s=180, # time to pass the USR2 signal to slurm before the job times out so that it can finish the run
                        tasks_per_node=1,
                        nodes=1,
                        cpus_per_task=30, #24
                        mem_per_cpu=4096,
                        job_name='smac_hpo',
                        slurm_gres=f'gpu:{1}'
       )

    return maximum_runtime

def plot_pareto(smac: AbstractFacade, incumbents: list[Configuration]) -> None:
    """Plots configurations from SMAC and highlights the best configurations in a Pareto front."""
    average_costs = []
    average_pareto_costs = []
    for config in smac.runhistory.get_configs():
        # Since we use multiple seeds, we have to average them to get only one cost value pair for each configuration
        average_cost = smac.runhistory.average_cost(config)

        if config in incumbents:
            average_pareto_costs += [average_cost]
        else:
            average_costs += [average_cost]

    # Let's work with a numpy array
    costs = np.vstack(average_costs)
    pareto_costs = np.vstack(average_pareto_costs)
    pareto_costs = pareto_costs[pareto_costs[:, 0].argsort()]  # Sort them

    costs_x, costs_y = costs[:, 0], costs[:, 1]
    pareto_costs_x, pareto_costs_y = pareto_costs[:, 0], pareto_costs[:, 1]

    plt.scatter(costs_x, costs_y, marker="x", label="Configuration")
    plt.scatter(pareto_costs_x, pareto_costs_y, marker="x", c="r", label="Incumbent")
    plt.step(
        [pareto_costs_x[0]] + pareto_costs_x.tolist() + [np.max(costs_x)],  # We add bounds
        [np.max(costs_y)] + pareto_costs_y.tolist() + [np.min(pareto_costs_y)],  # We add bounds
        where="post",
        linestyle=":",
    )

    plt.title("Pareto-Front")
    plt.xlabel(smac.scenario.objectives[0])
    plt.ylabel(smac.scenario.objectives[1])
    plt.legend()
    global log_file_name
    # Save the plot to a file
    plt.savefig(f"pareto_front_{log_file_name}.png", format='png', dpi=300)  # Save as PNG with high resolution
    plt.show()
    
    
class SuccessiveHalvingCustom:
    def __init__(self, config_space, min_budget, max_budget, logger, reduction_factor=2, n_initial=1, seed=42):
        """
        Initialize the Successive Halving algorithm.

        Args:
        - config_space: A configuration space object from which configurations can be sampled.
        - min_budget (int): Minimum budget (e.g., minimum number of epochs or trials) per configuration.
        - max_budget (int): Maximum budget per configuration.
        - reduction_factor (int): How aggressively the algorithm reduces the number of configurations.
        """
        self.config_space = config_space  # Config space to sample from
        self.min_budget = min_budget  # Minimum budget for any configuration
        self.max_budget = max_budget  # Maximum budget for any configuration
        self.reduction_factor = reduction_factor  # Controls the halving rate
        self.rung = 0  # Keeps track of the current rung in the halving process
        self.current_configs = []  # Configurations being evaluated in the current rung
        self.results = []  # Stores the results of each evaluated configuration in the current rung
        self.evaluations = {}  # Dictionary to store the final performance of each configuration across all rounds
        self.config_history = []  # Stores every configuration that has been evaluated along with results
        ckpt_name = ''
        for hp in config_space.get_hyperparameters():
            if isinstance(hp, Constant) and hp.name != 'model':
                ckpt_name += '_' + hp.name.split('_')[0][0] + hp.name.split('_')[1][0]
        self.checkpoint_path = 'successive_halving_checkpoint' + ckpt_name + '.pkl'
        
        # Calculate the number of rungs (stages) in successive halving
        self.num_rungs = int(math.log(max_budget / min_budget, reduction_factor)) + 1

        self.num_initial_configs = max(int(self.reduction_factor ** (self.num_rungs - 1)), n_initial)
        # Initial budget starts with the minimum budget
        self.current_budget = self.min_budget
        self.seed = seed
        # Track total number of configurations sampled
        self.total_configs_sampled = 0
        
        self.logger = logger
        
        self.logger.info(f"\t== SHA ConfigSpace ==\n\t{print_with_tabs(config_space,1)}")
        self.logger.info(f"\t eta: {args.eta}")
        self.logger.info(f"\t min_budget: {min_budget}")
        self.logger.info(f"\t max_budget: {max_budget}")
        self.logger.info(f"\t num_rungs: {self.num_rungs}")

    
    def ask(self):
        """
        Return the next configuration to evaluate with the current budget.
        If the current rung has remaining configs, use those, otherwise sample new ones.
        """
        # If we have no more configs for this rung, sample new ones
        if not self.current_configs:
            num_configs = self._num_configs_for_rung(self.rung)
            self.current_configs = [self.config_space.sample_configuration() for _ in range(num_configs)]
            self.total_configs_sampled += num_configs
        
        # Return the next configuration to evaluate
        config = self.current_configs[0]
        return SimpleNamespace(config = config, budget = self.current_budget, seed = self.seed)

    def tell(self, config, result, exception=""):
        """
        Receive the result of a configuration's evaluation.

        Args:
        - config: The configuration evaluated.
        - result: The result (e.g., performance) of the configuration.
        """
        # Store the result for this configuration
        self.results.append((config, result))
        
        # Record every evaluated config in the history (with budget)
        self.config_history.append({'config': config, 'budget': self.current_budget, 'result': result, 'exception': exception})
        
        # Update the performance of the config (only keep the best result per config)
        if config not in self.evaluations or self.evaluations[config] > result:
            self.evaluations[config] = result
            
        self.current_configs.pop(0)
        # If all configurations in the current rung are evaluated, proceed to the next rung
        if len(self.results) == self._num_configs_for_rung(self.rung):
            self._advance_rung()
    
    def _advance_rung(self):
        """
        Advance to the next rung, retaining only the top-performing configurations.
        """
        self.rung += 1
        if self.rung >= self.num_rungs:
            print("Successive Halving completed.")
            return
        
        # Sort the configurations based on their results and keep the top fraction
        self.results.sort(key=lambda x: x[1], reverse=False)
        num_survivors = len(self.results) // self.reduction_factor
        self.current_configs = [config for config, _ in self.results[:num_survivors]]
        self.results = []
        
        # Increase the budget for the next rung
        self.current_budget = self.min_budget * (self.reduction_factor ** self.rung)

    def _num_configs_for_rung(self, rung):
        """
        Calculate the number of configurations for a given rung.
        """
        return max(1, self.num_initial_configs // (self.reduction_factor ** rung))

    def is_done(self):
        """
        Check if the algorithm has finished evaluating all configurations.
        """
        return self.rung >= self.num_rungs

    def save_state(self):
        """
        Save the current state of the algorithm to a checkpoint file.
        """
        state = {
            'config_space': self.config_space,
            'min_budget': self.min_budget,
            'max_budget': self.max_budget,
            'reduction_factor': self.reduction_factor,
            'rung': self.rung,
            'current_configs': self.current_configs,
            'results': self.results,
            'evaluations': self.evaluations,
            'config_history': self.config_history,
            'current_budget': self.current_budget,
            'total_configs_sampled': self.total_configs_sampled,
            'num_rungs': self.num_rungs,
            'seed': self.seed,
            'logger': self.logger,
            'num_initial_configs': self.num_initial_configs
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self):
        """
        Load the state of the algorithm from a checkpoint file.
        """
        try:
            with open(self.checkpoint_path, 'rb') as f:
                state = pickle.load(f)
            
            assert state['config_space'] == self.config_space
            assert state['num_initial_configs'] == self.num_initial_configs
            assert state['min_budget'] == self.min_budget
            assert state['max_budget'] == self.max_budget
            assert state['reduction_factor'] == self.reduction_factor
            assert state['num_rungs'] == self.num_rungs
            assert state['rung'] <= self.num_rungs
            assert state['current_budget'] == self.min_budget * (self.reduction_factor ** state['rung'])
            
            self.config_space = state['config_space']
            self.min_budget = state['min_budget']
            self.max_budget = state['max_budget']
            self.reduction_factor = state['reduction_factor']
            self.num_rungs = state['num_rungs']
            self.rung = state['rung']
            self.current_configs = state['current_configs']
            self.results = state['results']
            self.evaluations = state['evaluations']
            self.config_history = state['config_history']
            self.current_budget = state['current_budget']
            self.total_configs_sampled = state['total_configs_sampled']
            self.seed = state['seed']
            self.logger = state['logger']
            print("State loaded successfully.")
        except FileNotFoundError:
            print("Checkpoint file not found. Starting from scratch.")
        except AssertionError:
            print("Checkpoint file does not match the current configuration. Starting from scratch.")
    
    def get_best_config(self):
        """
        Get the best configuration found so far.
        """
        if not self.evaluations:
            return None
        return max(self.evaluations.items(), key=lambda x: x[1])[0]

    def get_config_history(self):
        """
        Get the history of all configurations evaluated with their results.
        """
        return self.config_history
    
def smac_object(args, cs, logger):
    
    logger.info(f"\t== SMAC ConfigSpace ==\n\t{print_with_tabs(cs,1)}")
    if args.multiobjective:
        scenario = Scenario(
                    cs,
                    name="SMAC_trial_w_embedding",
                    objectives=["val_loss", "train_time"],
                    walltime_limit=23*60*60, 
                    n_trials=args.n_trials, # Evaluate max 500 different trials
                    min_budget=20*60,  # Train the MLP using a hyperparameter configuration for at least 5 epochs
                    max_budget=23*60*60,  # Train the MLP using a hyperparameter configuration for at most 25 epochs
                    n_workers=1,
                    seed=0,
                    deterministic=True
        )
    else:
        scenario = Scenario(
            cs,
            name="SMAC_trial_w_embedding",
            walltime_limit=23*60*60,
            n_trials=args.n_trials, # Evaluate max 500 different trials
            min_budget=20*60, # Train the MLP using a hyperparameter configuration for at least 5 epochs
            max_budget=23*60*60,  # Train the MLP using a hyperparameter configuration for at most 25 epochs
            n_workers=1,
            seed=0,
            deterministic=True
        )

    logger.info(f"\t{print_with_tabs(scenario,1)}")

    if args.surrogate == "gp":
        kernel = MaternKernel(nu=2.5) * ConstantKernel(1.0, constant_value_bounds="fixed")  # Radial Basis Function (RBF) kernel
        model = GaussianProcess(configspace=cs, kernel=kernel)
    else:
        model = RandomForest(configspace=cs)
        
    multi_objective_algorithm = ParEGO(scenario)

    initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=args.n_initial)
    if args.multiobjective:
        intensifier = SuccessiveHalving(scenario, eta=args.eta, incumbent_selection="highest_budget")
    else:
        intensifier = SuccessiveHalving(scenario, eta=args.eta)
    
    logger.info("\t== Creating SMAC MultiFidelityFacade ==")

    if args.multiobjective:
        smac = MultiFidelityFacade(
            scenario=scenario,
            target_function=Trainer.train,
            initial_design=initial_design,
            intensifier=intensifier,
            multi_objective_algorithm=multi_objective_algorithm,
            overwrite=False,            
            model=model,
        )
    else:
        smac = MultiFidelityFacade(
            scenario=scenario,
            target_function=Trainer.train,
            initial_design=initial_design,
            intensifier=intensifier,
            overwrite=False,            
            model=model,
        )

    logger.info(f"\t{print_with_tabs(smac, 1)}")
    logger.info(f"\t== SMAC Configuration ==\n\t{print_with_tabs(smac._scenario,1)}")
    logger.info(f"\t== SMAC Configuration ==\n\t{print_with_tabs(smac._intensifier,1)}")
    logger.info(f"\t== SMAC Configuration ==\n\t{print_with_tabs(smac._multi_objective_algorithm,1)}")
    logger.info(f"\t== SMAC Configuration ==\n\t{print_with_tabs(smac._model,1)}")

    logger.info(f"\t Multi-objective optimization: {args.multiobjective}")
    logger.info(f"\t Number of initial configurations: {args.n_initial}")
    logger.info(f"\t Number of trials: {args.n_trials}")
    logger.info(f"\t eta: {args.eta}")
    logger.info("\t== Starting the optimization ==")
    
    return smac
    
def main_smac(args):
    
    setup_logger('HPO_gpt2')
    logger = logging.getLogger('HPO_gpt2')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"  
    logger.info(f'============Starting============\n')   
    
    # Define the configuration space
    cs = ConfigurationSpace()
    model = Constant("model", value="custom")
    learning_rate = Constant("learning_rate", value=7.4533e-05) if args.lr_constant else UniformFloatHyperparameter("learning_rate", 1e-5, 1e-3, default_value=1e-4, log=True) ## originally 1e-6 to 1e-4
    weight_decay = Constant("weight_decay", value=0.008) if args.wd_constant else UniformFloatHyperparameter("weight_decay", 1e-6, 0.1, default_value=0.01, log=True)
    sequence_length = Constant("sequence_length", value=1024) if args.sl_constant else UniformIntegerHyperparameter("sequence_length", 256, 1024, default_value=1024)
    batch_size = CategoricalHyperparameter("batch_size", [1, 2, 4, 8, 16], default_value=4)
    n_head = CategoricalHyperparameter("n_head", [4, 6, 8, 10, 12], default_value=6)
    n_layer = CategoricalHyperparameter("n_layer", [4, 6, 8, 10, 12], default_value=6)
    n_embd = CategoricalHyperparameter("n_embd", [240, 480, 720, 960, 1200], default_value=480)
    # n_embd = CategoricalHyperparameter("n_embd", [256, 384, 512, 768, 1024], default_value=384)
    
    # warmup_iters = Constant("warmup_iters", value=700) if args.warmup_constant else UniformIntegerHyperparameter("warmup_iters", 500, 1000, default_value=700)
    # warmup_time = Constant("warmup_time", value=60*60) if args.warmup_constant else UniformIntegerHyperparameter("warmup_time", 10*60, 60*60, default_value=60*60)
    # learning_rate_decay_frac = Constant("learning_rate_decay_frac", value=0.0) if args.lr_decay_constant else UniformFloatHyperparameter("learning_rate_decay_frac", 0.0, 0.2, default_value=0.0)
    # cosine_restarts = Constant("cosine_restarts", value=0) if args.cosine_constant else UniformIntegerHyperparameter("cosine_restarts", 0, 4, default_value=0)
    
    # cs.add_hyperparameters([learning_rate, weight_decay, sequence_length, batch_size, n_head, n_layer, n_embd, model, 
    #                         warmup_iters, warmup_time, learning_rate_decay_frac, cosine_restarts])

    cs.add_hyperparameters([learning_rate, weight_decay, sequence_length, batch_size, n_head, n_layer, n_embd, model])
    
    print(f"{'using SMAC for optimization' if args.smac else 'using Successive Halving for optimization'}")
    
    if args.smac:
        sha = smac_object(args, cs, logger)
    else:
        sha = SuccessiveHalvingCustom(cs, min_budget=20*60, max_budget=23*60*60, logger=logger, reduction_factor=args.eta, n_initial=100)
        if args.checkpoint:
            sha.load_state()
    
    while not sha.is_done():
        info = sha.ask()
        assert info.seed is not None
        # print(info)
        experiment = Trainer(info.config, budget=info.budget, seed=info.seed, logger_=logger, max_budget=23*60*60, lr_schedule_time=True)
        try:
            cost = Trainer.train(experiment)
            exception = ""
        except Exception as e:
            print(f"Exception: {e}")
            cost = np.inf
            exception = str(e)
        value = TrialValue(cost=cost, time=0.5) if args.smac else cost
        
        if args.smac:
            sha.tell(info.config, value)
        else:
            sha.tell(info.config, value, exception)
            sha.save_state()

    incumbents = sha.intensifier.get_incumbents()  if args.smac else sha.get_best_config()
    # Print the best configuration
    print(f"Best found configuration: {incumbents}")
    
    global log_file_name
    pickle.dump(incumbents, open(f"incumbents_w_embedding.pkl", "wb"))
    pickle.dump(sha, open(f"sha_w_embedding.pkl", "wb"))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("-s", "--slurm", type=bool, default=False, help="flag to run training on slurm") # if not provided you can just run it from terminal (for debugging)
    parser.add_argument('-i', "--input_bin", type=str, default="dev/data/fineweb10B/fineweb_train_*.bin", help="input .bin to train on")
    parser.add_argument('-j', "--input_val_bin", type=str, default="dev/data/fineweb10B/fineweb_val_*.bin", help="input .bin to eval validation loss on")
    parser.add_argument('-o', "--output_dir", type=str, default="", help="output directory to which to write logs and checkpoints")
    parser.add_argument('-e', "--model_name", type=str, default="d6", help="gpt2-tiny|gpt2|gpt2-medium|gpt2-large|gpt2-xl|d6|d12|d24|d36|d48")
    parser.add_argument('-n', "--checkpoint_every", type=int, default=0, help="save a checkpoint every N steps")
    # token layout for each step of the optimization
    parser.add_argument('-b', "--batch_size", type=int, default=4, help="batch size, in units of #batch dimensions")
    parser.add_argument('-t', "--sequence_length", type=int, default=1024, help="sequence length")
    parser.add_argument('-d', "--total_batch_size", type=int, default=-1, help="total desired batch size, in units of #tokens")
    parser.add_argument('-nh', "--n_head", type=int, default=12, help="number of attention heads")
    parser.add_argument('-nl', "--n_layer", type=int, default=12, help="number of layers")
    parser.add_argument('-ne', "--n_embd", type=int, default=768, help="number of embeddings")
    # workload (number of steps)
    parser.add_argument('-v',"--val_loss_every", type=int, default=0, help="every how mant steps to evaluate val loss?")

    # parser.add_argument('-x', "--num_iterations", type=int, default=-1, help="number of iterations to run")
    # optimization
    parser.add_argument('-l', "--learning_rate", type=float, default=1e-4, help="learning rate warmup iterations")
    parser.add_argument('-u', "--warmup_iters", type=int, default=700, help="learning rate warmup iterations")
    parser.add_argument('-q', "--learning_rate_decay_frac", type=float, default=0.0, help="learning rate warmup iterations")
    parser.add_argument('-c', "--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument('-m', "--val_max_steps", type=int, default=500, help="how many batches of val to average?")
    parser.add_argument("--dtype", type=str, default="float32", help="float32|float16|bfloat16")
    parser.add_argument('-z', "--zero_stage", type=int, default=1, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.add_argument("--multiobjective", type=bool, default=False, help="multiobjective optimization")
    parser.add_argument("--n_initial", type=int, default=5, help="number of initial configurations to evaluate")
    parser.add_argument("--n_trials", type=int, default=500, help="number of trials to evaluate")
    parser.add_argument("--eta", type=int, default=2, help="eta parameter for Hyperband")
    parser.add_argument("--surrogate", type=str, default="gp", help="surrogate model to use")
    parser.add_argument("--smac", type=bool, default=False, help="use SMAC for optimization")
    parser.add_argument("--checkpoint", type=bool, default=True, help="load checkpoint")
    parser.add_argument("--lr_constant", type=bool, default=False, help="constant learning rate")
    parser.add_argument("--wd_constant", type=bool, default=False, help="constant weight decay")
    parser.add_argument("--sl_constant", type=bool, default=False, help="constant sequence length")
    args = parser.parse_args()
    
    if args.slurm == True:
        print("Running on slurm")
        global ex
        global q
        maximum_runtime = 0
        log_folder = './logs_cluster/'
        maximum_runtime = set_queue('mlhiwi', log_folder)
        submit_func = ex.submit
        job = submit_func(main_smac, args)

        print(job)
    else:
        print("Running on local machine")
        print(args.slurm)
        main_smac(args)
