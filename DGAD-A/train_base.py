import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
import pickle
import csv
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from diffab.datasets import get_dataset
from diffab.models import get_model
from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.train import *


from torch.utils.data import Dataset
import wandb
import random
import wandb
import datetime
import copy
import lmdb

from diffab.models.diffab import detach_all
from diffab.datasets.sampling import BalancedClusterSampler, Tracker

#
# UTILS
#

def setup_and_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='./configs/train/codesign_single_base.yml')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--finetune', type=str, default=None)

    #Dgad sepcific
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--aux_dataset_dir', type=str, default=None)
    parser.add_argument('--aux_dataset_name', type=str, default=None)
    parser.add_argument('-ra','--reset_aux_dataset', type=bool, default=None)
    parser.add_argument('-sb', '--sampling_balance', type=float, default=None)
    parser.add_argument('-gs', '--guidance_strength', type=float, default=None)
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--sampling_mixing', type=str, default=None)
    parser.add_argument('--sampling_multiplier', type=int, default=None)

    args = parser.parse_args()
    return args

def update_config_from_args(config, args):
    """
    Update config with args using dot notation mapping.
    Args take precedence over config values.
    """

    # Define mapping from arg names to config paths
    arg_to_config_map = {
        'experiment_name': 'dgad.experiment_name',
        'aux_dataset_dir': 'dgad.aux_dataset.aux_dataset_dir',
        'reset_aux_dataset': 'dgad.aux_dataset.reset_aux_dataset',
        'sampling_balance': 'dgad.sampler.sampling_balance',
        'guidance_strength': 'dgad.guidance_strength',
        'n_samples': 'dgad.aux_dataset.n_samples',
        'sampling_mixing': 'dgad.sampler.sampling_mixing',
        'sampling_multiplier': 'dgad.sampler.sampling_multiplier',
    }
    
    def set_nested_value(config_dict, path, value):
        """Set value in nested dict using dot notation path."""
        keys = path.split('.')
        current = config_dict
        
        # Navigate to parent dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    # Update config with non-None args
    for arg_name, config_path in arg_to_config_map.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            print(f"Overriding config {config_path} with arg value: {arg_value}")
            set_nested_value(config, config_path, arg_value)
    
    return config


#
# WANDB  CONFIG
#


def configure_wandb(config, experiment_name=None):
    print('Configuring wandb...')
    print(config)
    
    if experiment_name is None:
        # Use a default name if not provided
        experiment_name = os.environ.get('EXPERIMENT_NAME', 'no_name')

    experiment_time_id = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    experiment_id = experiment_name + '__' + experiment_time_id

    hp=copy.deepcopy(config)

    run = wandb.init(
            entity="simon-vermeir-ugent-universiteit-gent",
            project="Thesis_DGAD",
            id=f"{experiment_id}",
            config=hp
           )
    
    print(hp)

    return run


#
# TRAINING
#


def train(dataloader,val_dataloader, sampler, tracker, model, optimizer, scheduler, args, config, run, ckpt_dir):

    for epoch in tracker.epochs(100):
        sampler.set_epoch(epoch)
        for batch in tracker(dataloader):  # Tracker wraps the dataloader!
    
            # Train
            # - - - - - - - - - - - - - -

            model.train()
            time_start = current_milli_time()

            # Prepare batch
            batch = recursive_to(batch, args.device)

            # Forward pass
            loss_dict = model(batch)
            #print(loss_dict)
            loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
            loss_dict['overall'] = loss
            time_forward_end = current_milli_time()

        
            # Backward pass
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            time_backward_end = current_milli_time()
            

            # Logging
            # - - - - - - - - - - - - - -


            # Wandb
            run.log({"iteration": tracker.step,
                     #"step": tracker.step,
                     "loss": loss_dict['overall'].item(), 
                     "loss_rot": loss_dict['rot'].item(),
                     "loss_pos": loss_dict['pos'].item(),
                     "loss_seq": loss_dict['seq'].item(),
                     "grad": orig_grad_norm,
                     "lr": optimizer.param_groups[0]['lr'],
                     "time_forward": (time_forward_end - time_start) / 1000,
                     "time_backward": (time_backward_end - time_forward_end) / 1000,
                     "time_total": (time_backward_end - time_start) / 1000,
                    }, step=tracker.step)
    
            if tracker.step % 1 == 0:
                tracker.print_progress(loss_dict)
            
            # Validation
            # - - - - - - - - - - - - - -
            #todo only validate after a complete epoch, or mark if it is a complete epoch
            validate(val_dataloader, sampler, tracker, model, optimizer, scheduler, args, config, run, ckpt_dir)

            # Safety check
            # - - - - - - - - - - - - - -

            if not torch.isfinite(loss):
                print('NaN or Inf detected.')
                ckpt_path = os.path.join(log_dir, f'checkpoint_nan_{tracker.step}.pt')
                tracker.save_checkpoint(
                    model, optimizer, scheduler, config, ckpt_path,
                    extra={
                        'batch': recursive_to(batch, 'cpu'),
                    }
                )
                raise KeyboardInterrupt()

        # End of epoch    
        validate(val_dataloader, sampler, tracker, model, optimizer, scheduler, args, config, run, ckpt_dir, complete_epoch=True)

            

    pass


def validate(val_dataloader, sampler, tracker, model, optimizer, scheduler, args, config, run, ckpt_dir, complete_epoch=False):
    
    def validate_step(tracker, run):
        
        loss_tape = ValidationLossTape()
        
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(val_dataloader, desc='Validate', dynamic_ncols=True)):
                # Prepare data
                batch = recursive_to(batch, args.device)
                # Forward
                loss_dict = model(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                loss_dict['overall'] = loss

                run.log({
                    "iteration": tracker.step,
                    #"step": tracker.step,
                    "epoch": tracker.epoch,
                    "val_step": tracker.step,
                    "val_loss": loss_dict['overall'].item(), 
                }, step=tracker.step)

                loss_tape.update(loss_dict, 1)

        avg_loss = loss_tape.log(tracker.step, tag='val')

        # Trigger scheduler
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        return avg_loss



    if (tracker.step % config.train.val_freq == 0) or complete_epoch:
        avg_val_loss = validate_step(tracker, run)
        # update the current avg_val_loss in the tracker
        tracker.avg_val_loss = avg_val_loss
        if not args.debug:
            if complete_epoch:
                ckpt_name = f'val_complete_epoch_{tracker.epoch}_step_{tracker.step}.pt'
            else:
                ckpt_name = f'val_{tracker.step}.pt'
            # in a validation step we save the model
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            tracker.save_checkpoint(model, optimizer, scheduler, config, ckpt_path, complete_epoch=complete_epoch)

    


#
# GENERAL
#


if __name__ == '__main__':

    # Parse arguments
    args = setup_and_parse_args()
    # Load configs
    config, config_name = load_config(args.config)
    # Update config with args
    config = update_config_from_args(config, args)

    # Set global seed
    seed_all(config.train.seed)

     # Configure wandb
    run = configure_wandb(config, experiment_name=config.dgad.experiment_name)

    # Setup ckpt dir
    if args.resume:
        log_dir = os.path.dirname(os.path.dirname(args.resume))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config.dgad.experiment_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        print(f'Logging checkpoints to {log_dir}')


    # DATA
    # - - - - - - - - - - - - - - -

    #For training

    print('Loading dataset...')

    base_train_dataset = get_dataset(config.dataset.train)
    

    base_sampler = BalancedClusterSampler(base_train_dataset, 
                                     sampling_balance=config.dgad.sampler.sampling_balance, 
                                     sampling_mixing=config.dgad.sampler.sampling_mixing, 
                                     seed=config.train.seed,
                                     multiplier=30)

    base_train_dataloader = DataLoader(base_train_dataset, 
                            sampler=base_sampler, 
                            collate_fn=PaddingCollate(), 
                            batch_size=config.train.batch_size, 
                            num_workers=args.num_workers)
    print('Number of workers: %d' % args.num_workers)
    tracker = Tracker(base_train_dataloader, base_sampler, n_steps=config.train.max_iters)

    # - - 

    # For validation

    base_val_dataset = get_dataset(config.dataset.val)

    base_val_dataloader =  DataLoader(base_val_dataset, 
                                      batch_size=config.train.batch_size, 
                                      collate_fn=PaddingCollate(), 
                                      shuffle=False, 
                                      num_workers=0)


    print(f'Train {len(base_train_dataset)} | Val {len(base_val_dataset)}')



    # MODEL
    # - - - - - - - - - - - - - - - -

    # Get the model

    print('Building model...')
    base_model = get_model(config.model).to(args.device)
    
    base_optimizer = get_optimizer(config.train.optimizer, base_model)
    base_scheduler = get_scheduler(config.train.scheduler, base_optimizer)
    base_optimizer.zero_grad()
    
    print('Number of parameters: %d' % count_parameters(base_model))

    # Load from checkpoint if specified

    if args.resume is not None:
        ckpt_path = args.resume if args.resume is not None else args.finetune
        print(f'Resuming from checkpoint for base model: {ckpt_path}')
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        base_model.load_state_dict(ckpt['model'])
        print('Resuming optimizer states for base model...')
        base_optimizer.load_state_dict(ckpt['optimizer'])
        print('Resuming scheduler states for base model...')
        base_scheduler.load_state_dict(ckpt['scheduler'])

        tracker.resume(ckpt['iteration'], ckpt['epoch'], complete_epoch=ckpt.get('complete_epoch', False))


    # TRAIN
    # - - - - - - - - - - - - - - - -
    keyboard_interrupt = False
    try:
        train(
                base_train_dataloader,
                base_val_dataloader,
                base_sampler, 
                tracker,
                base_model,
                base_optimizer, 
                base_scheduler,
                args,
                config,
                run,
                ckpt_dir
            )

    except KeyboardInterrupt:
        print('Terminating...')
        keyboard_interrupt = True
        

    # Save model at end of training
    ckpt_name = f'base_final_{tracker.step}.pt'
    if keyboard_interrupt:
        ckpt_name = f'base_final_keyboard_interrupt_{tracker.step}.pt'
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    tracker.save_checkpoint(base_model, base_optimizer, base_scheduler, config, ckpt_path, complete_epoch=True)

    #Finish run
    print('Finishing aux dataset build')
     # Finish the run and upload any remaining data.
    run.finish()