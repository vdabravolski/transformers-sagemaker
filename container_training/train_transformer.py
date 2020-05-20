import logging 
import json
import os
import importlib.util
import torch.distributed.launch as launch # TODO: uncomment it after local testing
import sys
from argparse import ArgumentParser

logger = logging.getLogger(__name__)


BOOL_FLAGS = ["--fp16", "--do_train", "--do_eval"]

def _arg_flags_to_bool(args):
    """
    Sagemaker doesn't allow to provide boolena flags in its hyperparameters dict.
    We need to convert params with. string value to bool flag as it's required by Transformer algorithms, e.g.:
        Sagemaker format   ->      Transformer format
        "fp16" : "true"    ->      --fp16

    This method converts some common flags.
    """
    
    converted_args = []
    
    for i, arg in enumerate(args):
        if (arg in BOOL_FLAGS):
            if args[i+1].lower()=="true":
                converted_args.append(arg)
            del args[i+1] #remove flag value            
        else:
            converted_args.append(arg)
            
    return converted_args
    

def get_training_world():

    """
    Calculates number of devices in Sagemaker distributed cluster
    """
    
    # Get params of Sagemaker distributed cluster from predefined env variables
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    num_cpus = int(os.environ["SM_NUM_CPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]
    
    # Define PyTorch training world
    world = {}
    world["number_of_processes"] = num_gpus if num_gpus > 0 else num_cpus
    world["number_of_machines"] = len(hosts)
    world["size"] = world["number_of_processes"] * world["number_of_machines"]
    world["machine_rank"] = hosts.index(current_host)
    world["master_addr"] = hosts[0]
    world["master_port"] = "55555" # port is defined by Sagemaker
    
    return world


def task_selector(sm_args, transformer_args):
    """
    This method finds a desired Transformer task, 
    augments transformer args with ones specific to Sagemaker training runtime,
    and returns training path and augmented argline.
    """
    
    # convert boolen flags
    transformer_args = _arg_flags_to_bool(transformer_args)
    
    if sm_args.nlp_problem.lower() == "language-modeling":        
        
        task_path = os.path.join(os.environ["SAGEMAKER_SUBMIT_DIRECTORY"], 
                                 "transformers/examples/language-modeling/run_language_modeling.py")
        
        if sm_args.dataset.lower() == "wiki":
            
            # Augment argline based on dataset
            transformer_args += ["--train_data_file", os.path.join(os.environ['SM_CHANNEL_TRAIN'], 'wiki.train.raw'),
                                 "--eval_data_file", os.path.join(os.environ['SM_CHANNEL_TEST'], 'wiki.test.raw'),
                                 "--output_dir", os.environ['SM_OUTPUT_DATA_DIR']]
        else:
            raise ValueError(f"Dataset {sm_args.dataset} is not supported.")
    else:
        raise ValueError(f"Task {sm_args.nlp_problem} is not supported.")
    
    return task_path, transformer_args


if __name__ == "__main__":
    
    # Get initial configuration to select appropriate HuggingFace task and its configuration
    print('Starting training...')
    parser = ArgumentParser()
    parser.add_argument('--nlp-problem', type=str, default="language-modeling", help="Define NLP problem to run from HuggingFace example library. See for options: \
                                                                               https://github.com/huggingface/transformers/tree/master/examples#the-big-table-of-tasks.")
    parser.add_argument('--dataset', type=str, default=None, help="Define which dataset to use.")
    sm_args, transformer_args = parser.parse_known_args()
    
    # Get task script and its cofiguration
    task_script, transformer_args = task_selector(sm_args, transformer_args)
    
    # Derive parameters of distributed training cluster in Sagemaker
    world = get_training_world()
    logger.info('Running \'{}\' backend on {} nodes and {} processes. World size is {}. Current host is {}'.
                format("NCCL", world["number_of_machines"], world["number_of_processes"], world["size"], world["machine_rank"]))

    # Creates launch configuration according to PyTorch Distributed Launch utility requirements: 
    # https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py
    launch_config = ["--nnodes", str(world['number_of_machines']), "--node_rank", str(world['machine_rank']),
                     "--nproc_per_node", str(world['number_of_processes']), "--master_addr", world['master_addr'], 
                     "--master_port", world['master_port']]
        
    # Launch distributed training. Note, that launch script configuration is passed as script arguments
    sys.argv = [""] + launch_config + [task_script]+ transformer_args
    print("***** sys.args *****")
    print(sys.argv)
    launch.main()
