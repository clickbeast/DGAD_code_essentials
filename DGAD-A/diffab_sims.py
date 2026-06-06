import torch
import torch.nn as nn

from diffab.modules.common.geometry import construct_3d_basis
from diffab.modules.common.so3 import rotation_to_so3vec
from diffab.modules.encoders.residue import ResidueEmbedding
from diffab.modules.encoders.pair import PairEmbedding
from diffab.modules.diffusion.dpm_full import FullDPM
from diffab.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom
from diffab.modules.common.geometry import reconstruct_backbone

from ._base import register_model


from diffab.modules.common.so3 import so3vec_to_rotation
from diffab.modules.common.geometry import reconstruct_backbone_partially



resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}


def detach_all(data):
    if isinstance(data, torch.Tensor):
        return data.detach()  # Detach the tensor
    elif isinstance(data, (list, tuple)):
        return type(data)(detach_all(x) for x in data)  # Recursively detach elements in lists/tuples
    elif isinstance(data, dict):
        return {key: detach_all(value) for key, value in data.items()}  # Recursively detach values in dictionaries
    else:
        return data  # Return non-tensor objects as-is



@register_model('diffab')
class DiffusionAntibodyDesign(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]
        self.residue_embed = ResidueEmbedding(cfg.res_feat_dim, num_atoms)
        self.pair_embed = PairEmbedding(cfg.pair_feat_dim, num_atoms)

        self.diffusion = FullDPM(
            cfg.res_feat_dim,
            cfg.pair_feat_dim,
            **cfg.diffusion,
        )

    def encode(self, batch, remove_structure, remove_sequence):
        """
        Returns:
            res_feat:   (N, L, res_feat_dim)
            pair_feat:  (N, L, L, pair_feat_dim)
        """
        # This is used throughout embedding and encoding layers
        #   to avoid data leakage.

        #
        #! ! ! ! 
        # Leakage opportonity here the other atoms will like info!!
        # -> nvm they are not consiedered i believe in the embedder
            #includes coord -> zerod
            #includes seq -> zerod
            #includes 

        context_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], 
            ~batch['generate_flag']     # Context means ``not generated''
        )

        structure_mask = context_mask if remove_structure else None
        sequence_mask = context_mask if remove_sequence else None

        res_feat = self.residue_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            fragment_type = batch['fragment_type'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
        )

        pair_feat = self.pair_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
        )

        R = construct_3d_basis(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )
        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        return res_feat, pair_feat, R, p
    
    def forward(self, batch):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = self.cfg.get('train_structure', True),
            remove_sequence = self.cfg.get('train_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        loss_dict = self.diffusion(
            v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res,
            denoise_structure = self.cfg.get('train_structure', True),
            denoise_sequence  = self.cfg.get('train_sequence', True),
        )
        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        batch, 
        sample_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = sample_opt.get('sample_structure', True),
            remove_sequence = sample_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']
        traj = self.diffusion.sample(v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, **sample_opt)
        return traj
    

    @torch.no_grad()
    def sample_for_aux_model(
        self, 
        batch, 
        sample_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = sample_opt.get('sample_structure', True),
            remove_sequence = sample_opt.get('sample_sequence', True)
        )


        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        step_0_batch = self.diffusion.sample_for_aux_model(v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, **sample_opt)
        
        #detach the last step to be sure.
        step_0_batch = detach_all(step_0_batch)        
        
        #Reconstruct batch with changes
        #step_0_batch = (v, s, a) 
        #aa -> ok -> s = aa
        #reconstruct_backbone. -> use this function to find back the positions
        v_out, p_out, s_out = step_0_batch
        
        #convert v_out to R_out
        #it is ued when only operating on one batch maybe
        R_out = so3vec_to_rotation(v_out) 

        batch['aa'] = s_out

        #Find the new heavy atom positions, attention: drops side chains
        #pos_atom_new : the new heavy atom positons
        #mask_atom_new: for new heavy atoms it will only indicate 4 first positions as valid. Since sidechains get thrown away.

        pos_atom_new, mask_atom_new = reconstruct_backbone_partially(
                #old heavy atom structure
                pos_ctx = batch['pos_heavyatom'],
                
                #generated
                R_new = R_out,
                t_new = p_out,
                aa = s_out,

                #copy
                chain_nb = batch['chain_nb'],
                res_nb = batch['res_nb'],
                mask_atoms = batch['mask_heavyatom'],
                mask_recons = batch['generate_flag'],
            )
        
        

        batch['pos_heavyatom'] = pos_atom_new
        batch['mask_heavy_atom'] = mask_atom_new

        return batch



    @torch.no_grad()
    def optimize(
        self, 
        batch, 
        opt_step, 
        optimize_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(
            batch,
            remove_structure = optimize_opt.get('sample_structure', True),
            remove_sequence = optimize_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        traj = self.diffusion.optimize(v_0, p_0, s_0, opt_step, res_feat, pair_feat, mask_generate, mask_res, **optimize_opt)
        return traj
