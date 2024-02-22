import os
import argparse

import torch
import numpy as np

import balsa
from balsa.util import plans_lib
from run import BalsaAgent


def main(args, balsa_params):
    print(args)
    print('---')
    print(balsa_params)

    agent = BalsaAgent(balsa_params)
    agent.curr_value_iter = 0
    agent.num_query_execs = 0
    agent.num_total_timeouts = 0
    agent.overall_best_train_latency = np.inf
    agent.overall_best_test_latency = np.inf
    agent.overall_best_test_swa_latency = np.inf
    agent.overall_best_test_ema_latency = np.inf
    agent.train_nodes = plans_lib.FilterScansOrJoins(agent.train_nodes)
    agent.test_nodes = plans_lib.FilterScansOrJoins(agent.test_nodes)

    train_ds, train_loader, _, val_loader = agent._MakeDatasetAndLoader()

    plans_dataset = train_ds.dataset if isinstance(
            train_ds, torch.utils.data.Subset) else train_ds
    
    model = agent._MakeModel(plans_dataset)
    model.load_state_dict(torch.load(args.model_checkpoint))
    print(f"Loaded model checkpoint from {args.model_checkpoint}")

    planner = agent._MakePlanner(model, plans_dataset)

    # Run TRAIN queries
    to_execute_test, execution_results_test = agent.PlanAndExecute(
        model, planner, is_test=False)
    
    # Run TEST queries
    to_execute_test, execution_results_test = agent.PlanAndExecute(
        model, planner, is_test=True)


if __name__ == '__main__':
    WANDB_MODE = 'disabled'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str)
    parser.add_argument('--run', type=str)
    args = parser.parse_args()

    name = args.run
    print(f"Loading Balsa for '{name}' experiment.")
    balsa_params = balsa.params_registry.Get(name)
    balsa_params.use_local_execution = True
    
    # No simulation is used for evaluating
    balsa_params.sim = None

    # Set timeout for training queries to 3 minutes
    balsa_params.initial_timeout_ms = 3 * 60 * 1000

    main(args, balsa_params)

# python test_model.py --run Balsa_JOBLeakageTest2 --model_checkpoint /app/balsa/wandb/run-20230704_175433-fed9n4a8/files/checkpoint.pt