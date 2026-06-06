import os
import argparse
import copy
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from diffab.datasets import get_dataset
from diffab.models import get_model
from diffab.modules.common.geometry import reconstruct_backbone_partially
from diffab.modules.common.so3 import so3vec_to_rotation
from diffab.utils.inference import RemoveNative
from diffab.utils.protein.writers import save_pdb
from diffab.utils.train import recursive_to
from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.transforms import *
from diffab.utils.inference import *

from diffab.models.diffab import DiffusionAntibodyDesign, DiffusionAntibodyDesignSims


def create_data_variants(config, structure_factory):
    structure = structure_factory()
    structure_id = structure['id']

    data_variants = []
    if config.mode == 'single_cdr':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            data_variants.append({
                'data': data_var,
                'name': f'{structure_id}-{cdr_name}',
                'tag': f'{cdr_name}',
                'cdr': cdr_name,
                'residue_first': residue_first,
                'residue_last': residue_last,
            })
    elif config.mode == 'multiple_cdrs':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        transform = Compose([
            MaskMultipleCDRs(selection=cdrs, augmentation=False),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        data_variants.append({
            'data': data_var,
            'name': f'{structure_id}-MultipleCDRs',
            'tag': 'MultipleCDRs',
            'cdrs': cdrs,
            'residue_first': None,
            'residue_last': None,
        })
    elif config.mode == 'full':
        transform = Compose([
            MaskAntibody(),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        data_variants.append({
            'data': data_var,
            'name': f'{structure_id}-Full',
            'tag': 'Full',
            'residue_first': None,
            'residue_last': None,
        })
    elif config.mode == 'abopt':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            for opt_step in config.sampling.optimize_steps:
                data_variants.append({
                    'data': data_var,
                    'name': f'{structure_id}-{cdr_name}-O{opt_step}',
                    'tag': f'{cdr_name}-O{opt_step}',
                    'cdr': cdr_name,
                    'opt_step': opt_step,
                    'residue_first': residue_first,
                    'residue_last': residue_last,
                })
    else:
        raise ValueError(f'Unknown mode: {config.mode}.')
    return data_variants

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int)
    parser.add_argument('-c', '--config', type=str, default='./configs/test/codesign_single.yml')
    parser.add_argument('-o', '--out_root', type=str, default='./results')
    parser.add_argument('-t', '--tag', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-gs', '--guidance_strength', type=float, default=1.0)
    parser.add_argument('--guidance_strength_per_component', type=str, default='False')
    parser.add_argument('--guidance_strength_pos', type=float, default=0.75)
    parser.add_argument('--guidance_strength_rot', type=float, default=0.5)
    parser.add_argument('--guidance_strength_seq', type=float, default=1.0)


    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(args.seed if args.seed is not None else config.sampling.seed)

    # Testset
    dataset = get_dataset(config.dataset.test)
    get_structure = lambda: dataset[args.index]

    # Logging
    structure_ = get_structure()
    structure_id = structure_['id']
    tag_postfix = '_%s' % args.tag if args.tag else ''
    log_dir = get_new_log_dir(os.path.join(args.out_root, config_name + tag_postfix), prefix='%04d_%s' % (args.index, structure_['id']))
    logger = get_logger('sample', log_dir)
    logger.info('Data ID: %s' % structure_['id'])
    data_native = MergeChains()(structure_)
    save_pdb(data_native, os.path.join(log_dir, 'reference.pdb'))

    # Load checkpoint and model
    # logger.info('Loading model config and checkpoints: %s' % (config.model.checkpoint_base))
    # base_ckpt = torch.load(config.model.checkpoint_base, map_location='cpu', weights_only=False)
    # base_cfg_ckpt = base_ckpt['config']
    # base_model = get_model(base_cfg_ckpt.model).to(args.device)
    # base_lsd = base_model.load_state_dict(base_ckpt['model'])
    # logger.info(str(base_lsd))

    logger.info(f'Config = {config}')

    # Load the base model

    logger.info('Loading base model config and checkpoints: %s' % (config.model.checkpoint_base))
    base_ckpt = torch.load(config.model.checkpoint_base, map_location='cpu', weights_only=False)
    base_cfg_ckpt = base_ckpt['config']
    base_model = DiffusionAntibodyDesign(base_cfg_ckpt.model).to(args.device)
    base_lsd = base_model.load_state_dict(base_ckpt['model'], strict=False)
    logger.info(str(base_lsd))

    # Load the aux model
   
    logger.info('Loading aux model config and checkpoints: %s' % (config.model.checkpoint_aux))
    aux_ckpt = torch.load(config.model.checkpoint_aux, map_location='cpu', weights_only=False)
    aux_cfg_ckpt = aux_ckpt['config']
    aux_model = DiffusionAntibodyDesign(aux_cfg_ckpt.model).to(args.device)
    aux_lsd = aux_model.load_state_dict(aux_ckpt['model'], strict=False)
    logger.info(str(aux_lsd))


    # Activate the sims model
    # Use epsilon_base and epsilon_aux to perform guidance ; use encoder_base and encoder_aux depending on epsilon.
    guidance_strength_per_component = True if args.guidance_strength_per_component.lower() == 'true' else False
    guidance_strength = args.guidance_strength
    if guidance_strength_per_component:
        guidance_strength = {
            'guidance_strength_pos': args.guidance_strength_pos,
            'guidance_strength_rot': args.guidance_strength_rot,
            'guidance_strength_seq': args.guidance_strength_seq,
        }
    sims_model = DiffusionAntibodyDesignSims(
        cfg = config.model,
        base_model = base_model,
        aux_model = aux_model,
        guidance_strength = guidance_strength,
        config = config,
        guidance_strength_per_component = guidance_strength_per_component,
    ).to(args.device)

    # Load the seq epsilon net no softmax weights
    sims_model.diffusion.base_eps_net.copy_eps_seq_net_weights_to_eps_seq_net_no_softmax()
    sims_model.diffusion.aux_eps_net.copy_eps_seq_net_weights_to_eps_seq_net_no_softmax()

    base_model.eval()
    aux_model.eval()
    sims_model.eval()


    # Make data variants
    data_variants = create_data_variants(
        config = config,
        structure_factory = get_structure,
    )

    # Save metadata
    metadata = {
        'identifier': structure_id,
        'index': args.index,
        'config': args.config,
        'items': [{kk: vv for kk, vv in var.items() if kk != 'data'} for var in data_variants],
    }
    
    
    with open(os.path.join(log_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Start sampling
    collate_fn = PaddingCollate(eight=False)
    inference_tfm = [ PatchAroundAnchor(), ]
    if 'abopt' not in config.mode:  # Don't remove native CDR in optimization mode
        inference_tfm.append(RemoveNative(
            remove_structure = config.sampling.sample_structure,
            remove_sequence = config.sampling.sample_sequence,
        ))
    inference_tfm = Compose(inference_tfm)

    for variant in data_variants:
        os.makedirs(os.path.join(log_dir, variant['tag']), exist_ok=True)
        logger.info(f"Start sampling for: {variant['tag']}")

        save_pdb(data_native, os.path.join(log_dir, variant['tag'], 'REF1.pdb'))       # w/  OpenMM minimization
    
        data_cropped = inference_tfm(
            copy.deepcopy(variant['data'])
        )
        data_list_repeat = [ data_cropped ] * config.sampling.num_samples
        loader = DataLoader(data_list_repeat, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
        count = 0
        for batch in tqdm(loader, desc=variant['name'], dynamic_ncols=True):
            torch.set_grad_enabled(False)
            #model.eval()
            
            batch = recursive_to(batch, args.device)
            
            if 'abopt' in config.mode:
                
                # Antibody optimization starting from native
                """ traj_batch = model.optimize(batch, opt_step=variant['opt_step'], optimize_opt={
                    'pbar': True,
                    'sample_structure': config.sampling.sample_structure,
                    'sample_sequence': config.sampling.sample_sequence,
                }) """
            else:
                
                print("performing sims")
                
                # De novo design
                """  
                    traj_batch = model.sample(batch, sample_opt={
                    'pbar': True,
                    'sample_structure': config.sampling.sample_structure,
                    'sample_sequence': config.sampling.sample_sequence,
                }) """
                
                ## ENTRY POINT FOR SIMS
                traj_batch = sims_model.sample_for_sims(batch, sample_opt={
                    'pbar': True,
                    'sample_structure': config.sampling.sample_structure,
                    'sample_sequence': config.sampling.sample_sequence,
                })
                
       
            aa_new = traj_batch[0][2]   # 0: Last sampling step. 2: Amino acid.
            pos_atom_new, mask_atom_new = reconstruct_backbone_partially(
                pos_ctx = batch['pos_heavyatom'],
                R_new = so3vec_to_rotation(traj_batch[0][0]),
                t_new = traj_batch[0][1],
                aa = aa_new,
                chain_nb = batch['chain_nb'],
                res_nb = batch['res_nb'],
                mask_atoms = batch['mask_heavyatom'],
                mask_recons = batch['generate_flag'],
            )


            aa_new = aa_new.cpu()
            pos_atom_new = pos_atom_new.cpu()
            mask_atom_new = mask_atom_new.cpu()

            for i in range(aa_new.size(0)):
                data_tmpl = variant['data']
                aa = apply_patch_to_tensor(data_tmpl['aa'], aa_new[i], data_cropped['patch_idx'])
                mask_ha = apply_patch_to_tensor(data_tmpl['mask_heavyatom'], mask_atom_new[i], data_cropped['patch_idx'])
                pos_ha  = (
                    apply_patch_to_tensor(
                        data_tmpl['pos_heavyatom'], 
                        pos_atom_new[i] + batch['origin'][i].view(1, 1, 3).cpu(), 
                        data_cropped['patch_idx']
                    )
                )


                save_path = os.path.join(log_dir, variant['tag'], '%04d.pdb' % (count, ))
                save_pdb({
                    'chain_nb': data_tmpl['chain_nb'],
                    'chain_id': data_tmpl['chain_id'],
                    'resseq': data_tmpl['resseq'],
                    'icode': data_tmpl['icode'],
                    # Generated
                    'aa': aa,
                    'mask_heavyatom': mask_ha,
                    'pos_heavyatom': pos_ha,
                }, path=save_path)



                # save_pdb({
                #     'chain_nb': data_cropped['chain_nb'],
                #     'chain_id': data_cropped['chain_id'],
                #     'resseq': data_cropped['resseq'],
                #     'icode': data_cropped['icode'],
                #     # Generated
                #     'aa': aa_new[i],
                #     'mask_heavyatom': mask_atom_new[i],
                #     'pos_heavyatom': pos_atom_new[i] + batch['origin'][i].view(1, 1, 3).cpu(),
                # }, path=os.path.join(log_dir, variant['tag'], '%04d_patch.pdb' % (count, )))
                count += 1

        logger.info('Finished.\n')


if __name__ == '__main__':
    main()
