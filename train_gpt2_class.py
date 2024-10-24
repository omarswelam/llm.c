
import time
import argparse
import tiktoken
import torch
import os 
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from train_gpt2 import GPT, GPTConfig, DistributedDataLoader, write_model
from contextlib import nullcontext
import torch._inductor.config as config
import math
import torch.distributed as dist
from ConfigSpace import Configuration, ConfigurationSpace, Float
import numpy as np
import pickle
import logging
import datetime
import wandb
import submitit


def setup_logger(name=None):

    if name is None:
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    current_time = datetime.datetime.now()
    formatted_date = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # File handler
    file_handler = logging.FileHandler(f'logs/smac_{formatted_date}.log')

    global log_file_name
    log_file_name = f'logs/smac_{formatted_date}'
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s',  datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)
    logger.propagate = False
    logging.captureWarnings(True)
    return logger

def print_with_tabs(obj, num_tabs=1):
    # Convert the object to a string representation
    obj_str = str(obj)

    # Split the string representation into lines
    lines = obj_str.split('\n')

    # first line should not be tabbed
    tabbed_lines = ["\t"*num_tabs +lines[0] + "\n"]

    # Add a tab at the beginning of each line
    tabbed_lines = tabbed_lines + ['\t'*(num_tabs+1) + line + "\n" for line in lines[1:]]

    # Join the tabbed lines back into a single string and print
    return "".join(tabbed_lines).rstrip("\n")

# setup_logger('train_eval_hpo')
logger = logging.getLogger('HPO_gpt2')

gpt_configs = {
    "d6": {"n_layer":6, "n_head":6, "n_embd":384},
    "d12": {"n_layer":12, "n_head":12, "n_embd":768},
    "d24": {"n_layer":24, "n_head":16, "n_embd":1024},
    "d36": {"n_layer":36, "n_head":20, "n_embd":1280},
    "d48": {"n_layer":48, "n_head":25, "n_embd":1600}
    }

import torch
import wandb
import time
import os
import numpy as np
import pickle
from contextlib import nullcontext

class Trainer:
    def __init__(self, config_space, seed, budget, max_budget,
                 input_bin: str = "dev/data/fineweb10B/fineweb_train_*.bin",
                 input_val_bin: str = "dev/data/fineweb10B/fineweb_val_*.bin",
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.0,
                 n_head: int = 12,
                 n_layer: int = 12,
                 n_embd: int = 768,
                 sequence_length: int = 1024,
                 batch_size: int = 8,
                 total_batch_size: int = -1,
                 warmup_iters: int = 700,
                 warmup_time: int = 60*60,
                 learning_rate_decay_frac: float = 0.0,
                 grad_clip: float = 1.0,
                 val_max_steps: int = 500,
                 dtype: str = "float32",
                 zero_stage: int = 1,
                 multi_objective: bool = False,
                 model_name: str = "d6",
                 logger_=None,
                 cosine_restarts: int = 0,
                 lr_schedule_time = False,
                 **kwargs):
        # Initialize parameters  
        print("config_space", config_space)
        print(type(config_space))      
        self.config_space = config_space
        self.model_name = config_space['model'] if "model" in config_space.keys() else model_name

        if self.model_name != "custom":
            for key in gpt_configs["d6"].keys():
                self.config_space[key] = gpt_configs[self.model_name][key]
                                    
        self.seed = seed
        self.budget = budget
        self.input_bin = input_bin
        self.max_budget = max_budget
        self.input_val_bin = input_val_bin
        self.batch_size = self.config_space["batch_size"] if "batch_size" in self.config_space.keys() else batch_size
        self.learning_rate = self.config_space["learning_rate"] if "learning_rate" in self.config_space.keys() else learning_rate
        self.weight_decay = self.config_space["weight_decay"] if "weight_decay" in self.config_space.keys() else weight_decay
        self.n_head = self.config_space["n_head"] if "n_head" in self.config_space.keys() else n_head
        self.n_layer = self.config_space["n_layer"] if "n_layer" in self.config_space.keys() else n_layer
        self.n_embd = self.config_space["n_embd"] if "n_embd" in self.config_space.keys() else n_embd
        self.sequence_length = self.config_space["sequence_length"] if "sequence_length" in self.config_space.keys() else sequence_length
        ####
        self.warmup_iters = self.config_space["warmup_iters"] if "warmup_iters" in self.config_space.keys() else warmup_iters
        self.warmup_time = self.config_space["warmup_time"] if "warmup_time" in self.config_space.keys() else warmup_time
        self.learning_rate_decay_frac = self.config_space["learning_rate_decay_frac"] if "learning_rate_decay_frac" in self.config_space.keys() else learning_rate_decay_frac
        self.cosine_restarts = self.config_space["cosine_restarts"] if "cosine_restarts" in self.config_space.keys() else cosine_restarts
        self.cosine_restarts += 1
        self.cosine_cycle_length = self.max_budget / self.cosine_restarts
        self.lr_schedule_time = lr_schedule_time
        ###
        self.grad_clip = grad_clip
        self.val_max_steps = val_max_steps
        self.dtype = dtype
        self.zero_stage = zero_stage
        self.multi_objective = multi_objective
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

        if str(self.model_name) == "custom":
            self.save_name = str(self.model_name)
            for key in self.config_space.keys():
                self.save_name += "_"
                for initial_letter in key.split("_"):
                    self.save_name += initial_letter
                self.save_name += f"_{self.config_space[key]}"
            
            print("save_name", self.save_name)
                # + f"_lr_{self.learning_rate}_wd_{self.weight_decay}_sl_{self.sequence_length}_bs_{self.batch_size}_" \
                # + f"h_{self.n_head}_l_{self.n_layer}_em_{self.n_embd}"
        else:
            self.save_name = str(self.model_name) \
                + f"_lr_{self.learning_rate}_wd_{self.weight_decay}_sl_{self.sequence_length}_bs_{self.batch_size}" 
                
        self.step = 0
        self.time_elapsed = 0.0
        # Initialize WandB

        # Setup logging, device, and random seeds
        self.logger = setup_logger() if logger_ is None else logger_
        self._setup_ddp()
        self._set_seeds()
        
        # Initialize model and data loaders
        self._init_model()
        self.train_loader, self.val_loader = self._init_dataloaders()
        
        self.tokens_per_fwdbwd = self.sequence_length * self.batch_size * self.ddp_world_size
        self.total_batch_size = total_batch_size if total_batch_size > 0 else self.tokens_per_fwdbwd
        assert self.total_batch_size % self.tokens_per_fwdbwd == 0, "total_batch_size must be a multiple of tokens_per_fwdbwd"
        self.run = self._init_wandb()
        
        self._log_setup()
        
        if os.path.exists("hp_details_w_embeddings.pkl"):
            with open("hp_details_w_embeddings.pkl", "rb") as f:
                self.hp_details = pickle.load(f)
        else:
            self.hp_details = []

        print("config_space", self.config_space)
        print("model_name", self.model_name)
        print("learning_rate", self.learning_rate)
        print("weight_decay", self.weight_decay)
        print("n_head", self.n_head)
        print("n_layer", self.n_layer)
        print("n_embd", self.n_embd)
        print("sequence_length", self.sequence_length)
        print("batch_size", self.batch_size)
        print("total_batch_size", self.total_batch_size)
        print("warmup_iters", self.warmup_iters)
        print("learning_rate_decay_frac", self.learning_rate_decay_frac)
        print("grad_clip", self.grad_clip)
        print("val_max_steps", self.val_max_steps)
        print("dtype", self.dtype)
        print("zero_stage", self.zero_stage)
        print("multi_objective", self.multi_objective)
        print("seed", self.seed)
        print("budget", self.budget)

    def _log_setup(self):
        section_tab = 3*"\t"
        print(f"Running train_eval_hpo with budget: {self.budget}, config_space: {self.config_space.get_dictionary() if isinstance(self.config_space, Configuration) else self.config_space}, seed: {self.seed}")
        self.logger.info(f"{section_tab}== Running train_eval_hpo ==")
        # log all the arguments
        self.logger.info(f"{section_tab}== Arguments:\n"
                    + section_tab + f"\tseed: {self.seed} \n"
                    + section_tab + f"\tbudget: {self.budget} \n"
                    + section_tab + f"\tinput_bin: {self.input_bin} \n"
                    + section_tab + f"\tinput_val_bin: {self.input_val_bin} \n"
                    + section_tab + f"\tmodel: {self.model_name} \n"
                    + section_tab + f"\tbatch_size: {self.batch_size} \n"
                    + section_tab + f"\ttokens_per_fwdbwd: {self.tokens_per_fwdbwd} \n"
                    + section_tab + f"\ttotal_batch_size: {self.total_batch_size} \n"
                    + section_tab + f"\twarmup_iters: {self.warmup_iters} \n"
                    + section_tab + f"\tlearning_rate_decay_frac: {self.learning_rate_decay_frac} \n"
                    + section_tab + f"\tgrad_clip: {self.grad_clip} \n"
                    + section_tab + f"\tval_max_steps: {self.val_max_steps} \n"
                    + section_tab + f"\tdtype: {self.dtype} \n"
                    + section_tab + f"\tzero_stage: {self.zero_stage} \n")
        
        self.logger.info(f"{section_tab}== configuration space:\n" 
            + section_tab + f"\tConfig_space:{self.config_space.get_dictionary() if isinstance(self.config_space, Configuration) else self.config_space} \n")
        
        self.logger.info(f"{section_tab}== Dataloaders setup:\n"
            + section_tab + f"\ttrain_loader.ntok_total: {self.train_loader.ntok_total} \n"
            + section_tab + f"\tval_loader.ntok_total: {self.val_loader.ntok_total} \n"
            + section_tab + f"\tval_max_steps: {self.val_max_steps} \n")
        
    def _init_wandb(self):
        return wandb.init(
            project="LLMs",
            entity="o-swelam",
            config={
                "config_space": self.config_space.get_dictionary() if isinstance(self.config_space, Configuration) else self.config_space,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "n_head": self.n_head,
                "n_layer": self.n_layer,
                "n_embd": self.n_embd,
                "sequence_length": self.sequence_length,
                "seed": self.seed,
                "budget": self.budget,
                "input_bin": self.input_bin,
                "input_val_bin": self.input_val_bin,
                "model": self.model_name,
                "batch_size": self.batch_size,
                "total_batch_size": self.total_batch_size,
                "warmup_iters": self.warmup_iters,
                "learning_rate_decay_frac": self.learning_rate_decay_frac,
                "grad_clip": self.grad_clip,
                "val_max_steps": self.val_max_steps,
                "dtype": self.dtype,
                "zero_stage": self.zero_stage,
                "multi_objective": self.multi_objective
            },
            name= self.save_name + "_seed_" + str(self.seed) + "_bud_" + str(self.budget),
        )

    def _set_seeds(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _load_model(self):
        if os.path.exists(f"./dev/models/{self.save_name}.pth"):
            print(f'loading previous training states from: {self.save_name}')
            ckpt = torch.load(f"./dev/models/{self.save_name}.pth", map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.step = ckpt['step']
            self.time_elapsed = ckpt['time_elapsed'] 

    def _init_model(self):
        # Initialize the model from config
        model_name = self.model_name
        if model_name[0] == "d" or model_name == "custom":
            model_config = {
                "custom": GPTConfig(block_size=1024, vocab_size=50257, n_layer=self.config_space['n_layer'],
                                    n_head=self.config_space['n_head'], n_embd=self.config_space['n_embd']),
                "d6": GPTConfig(block_size=1024, vocab_size=50257, n_layer=6, n_head=6, n_embd=384),
                "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
                "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
                "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
                "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
            }[model_name]
            self.model = GPT(model_config)
        else:
            self.model = GPT.from_pretrained(model_name)
        
        self.model.to(self.device)
        
        self.optimizer = self._get_optimizer()
        
        self._load_model()
        
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.device], output_device=self.device)
        
    def _setup_ddp(self):
        self.ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
        if self.ddp:
            # use of DDP atm demands CUDA, we set the device appropriately according to rank
            assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
            init_process_group(backend='nccl')
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0 # this process will do logging, checkpointing etc.
            self.seed_offset = 0 # each process gets the exact same seed
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.zero_stage = 0
            self.ddp_world_size = 1
            self.master_process = True
            self.seed_offset = 0
            # attempt to autodetect the device
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"

    def _init_dataloaders(self):
        train_loader = DistributedDataLoader(self.input_bin, self.batch_size, 
                                             self.sequence_length, process_rank=self.ddp_rank, num_processes=self.ddp_world_size)
        val_loader = DistributedDataLoader(self.input_val_bin, 1, 
                                           1024, process_rank=0, num_processes=1)
        return train_loader, val_loader

    def _get_optimizer(self):
        return self.model.configure_optimizers(weight_decay=self.weight_decay,
                                               learning_rate=self.learning_rate, betas=(0.9, 0.95),
                                               device_type=self.device, zero_stage=self.zero_stage)

    def _get_lr(self, step, num_iterations):
        min_lr = self.learning_rate * self.learning_rate_decay_frac
        if step < self.warmup_iters:
            return self.learning_rate * (step+1) / self.warmup_iters
        if step > num_iterations:
            return min_lr
        decay_ratio = (step - self.warmup_iters) / (num_iterations - self.warmup_iters)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        return min_lr + coeff * (self.learning_rate - min_lr)

        
    def _get_lr_time(self, time_spent, T_0):
        min_lr = self.learning_rate * self.learning_rate_decay_frac        
        if time_spent < self.warmup_time:
            return self.learning_rate * (time_spent + 0.001) / self.warmup_time

        time_spent_after_warmup = time_spent - self.warmup_time

        time_within_period = time_spent_after_warmup % T_0 

        decay_ratio = time_within_period / T_0
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))        
        lr = min_lr + coeff * (self.learning_rate - min_lr)
        return lr

    def _save_model(self, step, start_time, optimizer):
        ckpt = {
            'model_state_dict': self.model.state_dict() if not self.ddp else self.model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'time_elapsed': self.time_elapsed + time.time() - start_time,
            'step': step,
        }
        torch.save(ckpt, f"./dev/models/{self.save_name}.pth")

    def validate(self, num_steps=None):
        # Validation logic
        val_loss = 0.0
        self.model.eval()
        num_steps = num_steps if num_steps is not None else self.val_max_steps
        with torch.no_grad():
            for i in range(num_steps):
                x, y = self.val_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y, return_logits=False)
                val_loss += loss.item()
                if (i+1) % 20 == 0:
                    print(f"Validation step: {i}/{num_steps} | Loss: {val_loss / (i+1)}") 
        
        val_loss /= num_steps
        return val_loss

    @classmethod
    def train(cls, trainer):
        section_tab = 5*"\t"
        if trainer.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        
        trainer.logger.info(f"{section_tab}== Training ==\n")
        print(f"== Training ==")
        trainer.logger.info(f"{section_tab} device: {trainer.device}, dtype: {trainer.dtype}, ddprank: {trainer.ddp_rank}, ddpworldsize: {trainer.ddp_world_size}, ddp: {trainer.ddp}")
        print(f"device: {trainer.device}, dtype: {trainer.dtype}, ddprank: {trainer.ddp_rank}, ddpworldsize: {trainer.ddp_world_size}, ddp: {trainer.ddp}")
        ctx = torch.amp.autocast(device_type=trainer.device, dtype=trainer.ptdtype) if trainer.device == "cuda" else nullcontext()
        num_iterations = trainer.train_loader.ntok_total // trainer.total_batch_size
        grad_accum_steps = trainer.total_batch_size // trainer.tokens_per_fwdbwd
        
        trainer.validate(num_steps=1) # to make sure the memory works well under such conditions at max seq length
        timings = [0] * 200
        training_losses = [0] * 200
        start_time = time.time()
        while time.time() + trainer.time_elapsed - start_time < trainer.budget:
            trainer.step += 1
            t0 = time.time()
            trainer.model.train()
            trainer.optimizer.zero_grad(set_to_none=True)
            lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
            for micro_step in range(grad_accum_steps):
                # fetch a batch
                x, y = trainer.train_loader.next_batch()
                x, y = x.to(trainer.device), y.to(trainer.device)
                if trainer.ddp:
                    # we want only the last micro-step to sync grads in a DDP model
                    # the official way to do this is with model.no_sync(), but that is a
                    # context manager that bloats the code, so we just toggle this variable
                    trainer.model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
                # forward pass
                with ctx:
                    _, loss = trainer.model(x, y, return_logits=False)
                    # we have to scale the loss to account for gradient accumulation,
                    # because the gradients just add on each successive backward().
                    # addition of gradients corresponds to a SUM in the objective, but
                    # instead of a SUM we want MEAN, so we scale the loss here
                    loss = loss / grad_accum_steps
                    lossf += loss.detach() # keep track of the mean loss
                # backward pass
                loss.backward()

            if trainer.ddp:
                dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
            lossf = lossf.item()

            norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.grad_clip)
            lr = trainer._get_lr(trainer.step, num_iterations) if not trainer.lr_schedule_time else trainer._get_lr_time(trainer.time_elapsed + time.time() - start_time, trainer.cosine_cycle_length)
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = lr

            trainer.optimizer.step()
            
            # wait on the CPU for all device work to end so we get accurate per-iteration timings below
            if trainer.device == "mps":
                torch.mps.synchronize()
            elif trainer.device == "cuda":
                torch.cuda.synchronize()
            # time and print
            t1 = time.time()           

            training_losses.append(lossf)
            training_losses.pop(0)
            timings.append(t1-t0)
            timings.pop(0)

            if trainer.step % 200 == 0:
                trainer.logger.info(f"{section_tab} \tStep: {trainer.step}/{num_iterations} | Loss: {lossf} | LR: {lr} | Norm: {norm}")
                print(f"Step: {trainer.step}/{num_iterations} | Loss: {lossf} | LR: {lr} | Norm: {norm}")
                wandb.log({"step_loss":lossf, "avg train loss": np.mean(training_losses), "lr": lr, "step":trainer.step, "remaining_time": trainer.budget - (trainer.time_elapsed + time.time() - start_time)})
            
            if trainer.step % 1000 == 0:
                val_loss = trainer.validate(num_steps=100)
                trainer.logger.info(f"{section_tab} \tValidation loss: {val_loss}")
                print(f"Validation loss: {val_loss}")
                wandb.log({"val_loss": val_loss, "step": trainer.step})
                trainer._save_model(trainer.step, start_time, trainer.optimizer)
        
        trainer.logger.info(f"{section_tab}== Training complete ==\n")
        
        val_loss = trainer.validate()
        trainer.logger.info(f"{section_tab}Validation loss: {val_loss}")
        print(f"Validation loss: {val_loss}")
        wandb.log({"val_loss": val_loss, "step": trainer.step})
        
        trainer.hp_details.append({
            "budget": trainer.budget,
            "val_loss": val_loss, 
            "train_loss": np.mean(training_losses), "train_time": trainer.time_elapsed + time.time() - start_time, 
            "batch_size": trainer.batch_size, 
            "learning_rate": trainer.learning_rate,
            "weight_decay": trainer.weight_decay, 
            "sequence_length": trainer.sequence_length,
            "n_head": trainer.n_head,
            "n_layer": trainer.n_layer,
            "n_embd": trainer.n_embd,
                       })
        
        with open(f"hp_details_w_embeddings.pkl", "wb") as f:
            pickle.dump(trainer.hp_details, f)
        wandb.finish()
        
        if trainer.ddp:
            destroy_process_group()
            
        return val_loss
    
  
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
    # TODO: add main function to run the training for the non_HPO setup
    # TODO: Adjust the hp pickle file saves to accomodate for the different config_spaces
    # TODO: change the naming of models to accomodate for different config_spaces
    # TODO: adjust that config_space is actually split into the respective variables 
    

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
    args = parser.parse_args()
    
    trainer = Trainer(config_space = {}, budget=23*60*60, seed=42, **vars(args))
        
    if args.slurm == True:
        print("Running on slurm")
        global ex
        global q
        maximum_runtime = 0
        log_folder = './logs_cluster/'
        maximum_runtime = set_queue('mlhiwi', log_folder)
        submit_func = ex.submit
        job = submit_func(Trainer.train, trainer)

        print(job)
    else:
        print("Running on local machine")
        print(args.slurm)
        Trainer.train(trainer)