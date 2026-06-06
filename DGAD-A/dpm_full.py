import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm

from diffab.modules.common.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix
from diffab.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3
from diffab.modules.encoders.ga import GAEncoder
from .transition import RotationTransition, PositionTransition, AminoacidCategoricalTransition
from typing import TYPE_CHECKING
import numpy as np


if TYPE_CHECKING:
    from diffab.models.diffab import DiffusionAntibodyDesign

def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Args:
        R_pred: (*, 3, 3).
        R_true: (*, 3, 3).
    Returns:
        Per-matrix losses, (*, ).
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  # (ncol*3, )
    loss = loss.reshape(size + [3]).sum(dim=-1)    # (*, )
    return loss

def get_schedule(omega, omega_template):
    #omega = 1.0
    #omega_template = [(False,(10)),(True,(10,20,10)),(False, 50)]

    omega_schedule = np.array([])
    for entry in omega_template:
        if entry[0] == False:
            T = entry[1]
            # If T is a list (from YAML), extract the integer
            if isinstance(T, list):
                T = T[0]
            omega_schedule_part = omega * np.zeros(T)
            omega_schedule = np.concatenate((omega_schedule, omega_schedule_part)).flatten()
        else:
            T1, T2, T3 = entry[1]
            T = T1 + T2 + T3
            omega_schedule_part = np.concatenate((np.linspace(0, omega , T1, endpoint=False),
                                        omega * np.ones(T2),
                                        np.linspace(omega, 0, T3, endpoint=False))).flatten()

            omega_schedule = np.concatenate((omega_schedule, omega_schedule_part)).flatten()


    print(omega_schedule.shape)
    print(omega_schedule)

    return omega_schedule


def lie_algebra_guidance_so3(base_v, aux_v, guidance_strength):
    """
    SO(3) guidance using Lie algebra, following Eade (2017)
    """
    # 1. Convert axis-angle vectors to rotation matrices
    base_R = so3vec_to_rotation(base_v)  # (N, L, 3, 3)
    aux_R = so3vec_to_rotation(aux_v)    # (N, L, 3, 3)
    
    # 2. Compute the relative rotation from aux to base
    #    This is the rotation that, when applied to aux_R, gives base_R
    rel_R = aux_R.transpose(-2, -1) @ base_R  # (N, L, 3, 3)
    
    # 3. Map this relative rotation to the tangent space (Lie algebra)
    #    This gives a 3D vector representing the "difference" in rotation
    rel_omega = rotation_to_so3vec(rel_R)     # (N, L, 3)
    
    # 4. Extrapolate in tangent space by (1 + guidance_strength)
    #    This moves further along the geodesic from aux_R toward base_R
    guided_omega = (1 + guidance_strength) * rel_omega  # (N, L, 3)
    
    # 5. Map back to SO(3) by exponentiating the tangent vector
    #    This gives the rotation matrix for the guided orientation
    guided_R = aux_R @ so3vec_to_rotation(guided_omega)  # (N, L, 3, 3)
    
    # 6. Convert back to axis-angle for downstream use
    guided_v = rotation_to_so3vec(guided_R)  # (N, L, 3)
    
    return guided_v


def lie_algebra_guidance_so3_anchored(base_v, aux_v, guidance_strength, R_t,
                             push_away=True, scale_match=True, clip_tau=None):
    """
    SO(3) guidance using Lie algebra, anchored at the CURRENT state R_t.

    Args:
        base_v: (N, L, 3)  axis-angle of base model's x0 prediction.
        aux_v:  (N, L, 3)  axis-angle of aux  model's x0 prediction.
        guidance_strength (float or tensor broadcastable to (N,L,1)): ω.
        R_t:    (N, L, 3, 3) rotation matrices of the CURRENT noisy state.
        push_away (bool): if True, do (1+ω) u_base − ω u_aux (subtract aux shift).
                          if False, do (1−ω) u_base + ω u_aux (blend toward aux).
        scale_match (bool): match ||u_aux|| to ||u_base|| before combining.
        clip_tau (float or None): if set, clip ||u_guided|| to <= clip_tau.

    Returns:
        guided_v: (N, L, 3) axis-angle of guided x0 prediction (like your original).
    """

    # 1) Convert axis-angle vectors to rotation matrices (predictions)
    base_R = so3vec_to_rotation(base_v)  # (N, L, 3, 3)
    aux_R  = so3vec_to_rotation(aux_v)   # (N, L, 3, 3)

    # 2) Express both predictions as displacements from the CURRENT state R_t
    #    (i.e., rays in the SAME tangent space at R_t)
    #    u_* are axis-angle vectors in T_{R_t}SO(3): u = log(R_t^T * R_pred)
    Rt_T = R_t.transpose(-2, -1)
    u_base = rotation_to_so3vec(Rt_T @ base_R)   # (N, L, 3)
    u_aux  = rotation_to_so3vec(Rt_T @ aux_R)    # (N, L, 3)

    # 3) Optional: scale-match aux to base so magnitudes are comparable
    if scale_match:
        nb = u_base.norm(dim=-1, keepdim=True)               # (N, L, 1)
        na = u_aux.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        u_aux = u_aux * (nb / na)

    # 4) Combine in the tangent at R_t
    #    push_away=True  -> subtract aux shift:  (1+ω)u_base - ω u_aux
    #    push_away=False -> blend toward aux:    (1-ω)u_base + ω u_aux
    omega = guidance_strength
    if push_away:
        u_guided = (1 + omega) * u_base - omega * u_aux
    else:
        u_guided = (1 - omega) * u_base + omega * u_aux

    # 5) Map back to SO(3) from the CURRENT state: R_guided = R_t * exp(u_guided)
    guided_R = R_t @ so3vec_to_rotation(u_guided)  # (N, L, 3, 3)

    # 6) Convert back to axis-angle for downstream use
    guided_v = rotation_to_so3vec(guided_R)        # (N, L, 3)
    return guided_v



def lie_algebra_guidance_so3_anchored_clipped(base_v, aux_v, guidance_strength, R_t):
    """
    SO(3) guidance using Lie algebra, anchored at the CURRENT state R_t.

    Args:
        base_v: (N, L, 3)  axis-angle of base model's x0 prediction.
        aux_v:  (N, L, 3)  axis-angle of aux  model's x0 prediction.
        guidance_strength (float or tensor broadcastable to (N,L,1)): ω.
        R_t:    (N, L, 3, 3) rotation matrices of the CURRENT noisy state.
        push_away (bool): if True, do (1+ω) u_base − ω u_aux (subtract aux shift).
                          if False, do (1−ω) u_base + ω u_aux (blend toward aux).
        scale_match (bool): match ||u_aux|| to ||u_base|| before combining.
        clip_tau (float or None): if set, clip ||u_guided|| to <= clip_tau.

    Returns:
        guided_v: (N, L, 3) axis-angle of guided x0 prediction (like your original).
    """

    # 1) Convert axis-angle vectors to rotation matrices (predictions)
    base_R = so3vec_to_rotation(base_v)  # (N, L, 3, 3)
    aux_R  = so3vec_to_rotation(aux_v)   # (N, L, 3, 3)

    # 2) Express both predictions as displacements from the CURRENT state R_t
    #    u_* are axis-angle vectors in T_{R_t}SO(3): u = log(R_t^T * R_pred)
    Rt_T = R_t.transpose(-2, -1)
    u_base = rotation_to_so3vec(Rt_T @ base_R)   # (N, L, 3)
    u_aux  = rotation_to_so3vec(Rt_T @ aux_R)    # (N, L, 3)

    # 4) Combine in the tangent at R_t
    omega = guidance_strength
    u_guided = (1 + omega) * u_base - omega * u_aux
    
    #4.0 clip: ‖u_guided‖ < (1 + ω) * ‖u_base‖.

    # u_base, u_guided : (N, L, 3)  axis-angle tangents at the current state R_t
    
    # 4.1) Compute base step length ‖u_base‖ for each (N,L) item.
    # .norm(dim=-1)     → sqrt(x^2 + y^2 + z^2) over the 3-vector
    # keepdim=True      → keeps shape as (N, L, 1) so it broadcasts cleanly later
    # .clamp_min(1e-8)  → replaces values < 1e-8 with 1e-8 (avoids divide-by-zero)
    nb = u_base.norm(dim=-1, keepdim=True).clamp_min(1e-8)   # (N, L, 1), radians

    # 4.2) Compute guided step length ‖u_guided‖ with the same guards/shapes.
    g  = u_guided.norm(dim=-1, keepdim=True).clamp_min(1e-8) # (N, L, 1), radians

    # 4.3) Build the target radius τ = (1 + ω) * ‖u_base‖.
    # - 'omega' can be a scalar or a tensor that broadcasts to (N, L, 1).
    # - Broadcasting: PyTorch auto-expands singleton dims to match (N, L, 1).
    tau = (1 + omega) * nb                                   # (N, L, 1), radians

    
    # 4.4) Compute per-vector scale s = min(1, τ / ‖u_guided‖).
    # - If ‖u_guided‖ <= τ  → τ/g ≥ 1 → s = 1 (no change).
    # - If ‖u_guided‖ >  τ  → τ/g < 1 → s < 1 (shrink u_guided to lie on radius τ).
    # .clamp(max=1.0) ensures we never *increase* the step (s ≤ 1).
    s = (tau / g).clamp(max=1.0)                              # (N, L, 1) in (0,1]

    # 4.5) Apply the scale per (N,L,3) tangent vector via broadcasting over the last dim.
    # Multiplication is elementwise; s (N,L,1) expands to (N,L,3).
    u_guided = u_guided * s

    # 5) Map back to SO(3) from the CURRENT state
    guided_R = R_t @ so3vec_to_rotation(u_guided)  # (N, L, 3, 3)

    # 6) Convert back to axis-angle for downstream use
    guided_v = rotation_to_so3vec(guided_R)        # (N, L, 3)
    return guided_v


class EpsilonNet(nn.Module):

    def __init__(self, res_feat_dim, pair_feat_dim, num_layers, encoder_opt={}):
        super().__init__()
        self.current_sequence_embedding = nn.Embedding(25, res_feat_dim)  # 22 is padding
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 2, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
        )
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, **encoder_opt)

        self.eps_crd_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_rot_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_seq_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 20), nn.Softmax(dim=-1) 
        )

        self.eps_seq_net_no_softmax = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 20)
        )

    def copy_eps_seq_net_weights_to_eps_seq_net_no_softmax(self):

        """
        Copies weights from eps_seq_net (with softmax) to eps_seq_net_no_softmax (without softmax).
        Ignores the softmax layer, which has no parameters.
        """
        # Get the state dict from the net with softmax
        src_state = self.eps_seq_net.state_dict()
        # Load into the net without softmax, allowing missing/unexpected keys
        self.eps_seq_net_no_softmax.load_state_dict(src_state, strict=False)




    def forward(self, v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res, output_denoised_logits=False):
        """
        Args:
            v_t:    (N, L, 3).
            p_t:    (N, L, 3).
            s_t:    (N, L).
            res_feat:   (N, L, res_dim).
            pair_feat:  (N, L, L, pair_dim).
            beta:   (N,).
            mask_generate:    (N, L).
            mask_res:       (N, L).
        Returns:
            v_next: UPDATED (not epsilon) SO3-vector of orietnations, (N, L, 3).
            eps_pos: (N, L, 3).
        """
        N, L = mask_res.size()
        R = so3vec_to_rotation(v_t) # (N, L, 3, 3)

        # s_t = s_t.clamp(min=0, max=19)  # TODO: clamping is good but ugly.
        res_feat = self.res_feat_mixer(torch.cat([res_feat, self.current_sequence_embedding(s_t)], dim=-1)) # [Important] Incorporate sequence at the current step.
        res_feat = self.encoder(R, p_t, res_feat, pair_feat, mask_res)

        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].expand(N, L, 3)
        in_feat = torch.cat([res_feat, t_embed], dim=-1)

        # Position changes
        eps_crd = self.eps_crd_net(in_feat)    # (N, L, 3)
        eps_pos = apply_rotation_to_vector(R, eps_crd)  # (N, L, 3)
        eps_pos = torch.where(mask_generate[:, :, None].expand_as(eps_pos), eps_pos, torch.zeros_like(eps_pos))

        # New orientation
        eps_rot = self.eps_rot_net(in_feat)    # (N, L, 3)
        U = quaternion_1ijk_to_rotation_matrix(eps_rot) # (N, L, 3, 3)
        R_next = R @ U
        v_next = rotation_to_so3vec(R_next)     # (N, L, 3)
        v_next = torch.where(mask_generate[:, :, None].expand_as(v_next), v_next, v_t)

        # New sequence categorical distributions
        if not output_denoised_logits:
            c_denoised = self.eps_seq_net(in_feat)  # Already softmax-ed, (N, L, 20)
        else:
            c_denoised = self.eps_seq_net_no_softmax(in_feat)  # Not softmax-ed, (N, L, 20)
        
        return v_next, R_next, eps_pos, c_denoised



class FullDPM(nn.Module):

    def __init__(
        self, 
        res_feat_dim, 
        pair_feat_dim, 
        num_steps, 
        eps_net_opt={}, 
        trans_rot_opt={}, 
        trans_pos_opt={}, 
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
    ):
        super().__init__()
        self.eps_net = EpsilonNet(res_feat_dim, pair_feat_dim, **eps_net_opt)
        self.num_steps = num_steps
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)

        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer('_dummy', torch.empty([0, ]))

    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p

    def forward(self, v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, denoise_structure, denoise_sequence, t=None):
        N, L = res_feat.shape[:2]
        if t == None:
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)
        p_0 = self._normalize_position(p_0)

        if denoise_structure:
            # Add noise to rotation
            R_0 = so3vec_to_rotation(v_0)
            v_noisy, _ = self.trans_rot.add_noise(v_0, mask_generate, t)
            # Add noise to positions
            p_noisy, eps_p = self.trans_pos.add_noise(p_0, mask_generate, t)
        else:
            R_0 = so3vec_to_rotation(v_0)
            v_noisy = v_0.clone()
            p_noisy = p_0.clone()
            eps_p = torch.zeros_like(p_noisy)

        if denoise_sequence:
            # Add noise to sequence
            _, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t)
        else:
            s_noisy = s_0.clone()

        beta = self.trans_pos.var_sched.betas[t]
        v_pred, R_pred, eps_p_pred, c_denoised = self.eps_net(
            v_noisy, p_noisy, s_noisy, res_feat, pair_feat, beta, mask_generate, mask_res
        )   # (N, L, 3), (N, L, 3, 3), (N, L, 3), (N, L, 20), (N, L)

        loss_dict = {}

        # Rotation loss
        loss_rot = rotation_matrix_cosine_loss(R_pred, R_0) # (N, L)
        loss_rot = (loss_rot * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['rot'] = loss_rot

        # Position loss
        loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction='none').sum(dim=-1)  # (N, L)
        loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['pos'] = loss_pos

        # Sequence categorical loss
        post_true = self.trans_seq.posterior(s_noisy, s_0, t)
        log_post_pred = torch.log(self.trans_seq.posterior(s_noisy, c_denoised, t) + 1e-8)
        kldiv = F.kl_div(
            input=log_post_pred, 
            target=post_true, 
            reduction='none',
            log_target=False
        ).sum(dim=-1)    # (N, L)
        loss_seq = (kldiv * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['seq'] = loss_seq

        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        v, p, s, 
        res_feat, pair_feat, 
        mask_generate, mask_res, 
        sample_structure=True, sample_sequence=True,
        pbar=False,
    ):
        """
        Args:
            v:  Orientations of contextual residues, (N, L, 3).
            p:  Positions of contextual residues, (N, L, 3).
            s:  Sequence of contextual residues, (N, L).
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            v_rand = random_uniform_so3([N, L], device=self._dummy.device)
            p_rand = torch.randn_like(p)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_rand, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_rand, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            s_rand = torch.randint_like(s, low=0, high=19)
            s_init = torch.where(mask_generate, s_rand, s)
        else:
            s_init = s

        traj = {self.num_steps: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x
        for t in pbar(range(self.num_steps, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj
    


    @torch.no_grad()
    def sample_for_aux_model(
        self, 
        v, p, s, 
        res_feat, pair_feat, 
        mask_generate, mask_res, 
        sample_structure=True, sample_sequence=True,
        pbar=False,
    ):
        """
        Args:
            v:  Orientations of contextual residues, (N, L, 3).
            p:  Positions of contextual residues, (N, L, 3).
            s:  Sequence of contextual residues, (N, L).
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            v_rand = random_uniform_so3([N, L], device=self._dummy.device)
            p_rand = torch.randn_like(p)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_rand, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_rand, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            s_rand = torch.randint_like(s, low=0, high=19)
            s_init = torch.where(mask_generate, s_rand, s)
        else:
            s_init = s

        traj = {self.num_steps: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x
        for t in pbar(range(self.num_steps, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            #traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.
            #Throw away prev step don't need it anymore
            del traj[t]

        #Only return the last step 
        return traj[0]





    @torch.no_grad()
    def optimize(
        self, 
        v, p, s, 
        opt_step: int,
        res_feat, pair_feat, 
        mask_generate, mask_res, 
        sample_structure=True, sample_sequence=True,
        pbar=False,
    ):
        """
        Description:
            First adds noise to the given structure, then denoises it.
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)
        t = torch.full([N, ], fill_value=opt_step, dtype=torch.long, device=self._dummy.device)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            # Add noise to rotation
            v_noisy, _ = self.trans_rot.add_noise(v, mask_generate, t)
            # Add noise to positions
            p_noisy, _ = self.trans_pos.add_noise(p, mask_generate, t)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_noisy, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_noisy, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            _, s_noisy = self.trans_seq.add_noise(s, mask_generate, t)
            s_init = torch.where(mask_generate, s_noisy, s)
        else:
            s_init = s

        traj = {opt_step: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=opt_step, desc='Optimizing')
        else:
            pbar = lambda x: x
        for t in pbar(range(opt_step, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj





class FullDPMSims(nn.Module):

    def __init__(
        self, 
        res_feat_dim, 
        pair_feat_dim, 
        num_steps, 
        eps_net_opt={}, 
        trans_rot_opt={}, 
        trans_pos_opt={}, 
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
        base_model: 'DiffusionAntibodyDesign' = None,
        aux_model: 'DiffusionAntibodyDesign' = None,
        guidance_strength = 1.0,
        config = None,
        guidance_strength_per_component = False
    ):


        super().__init__()
        
        #old
        self.eps_net = EpsilonNet(res_feat_dim, pair_feat_dim, **eps_net_opt)
        self.num_steps = num_steps
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)

        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer('_dummy', torch.empty([0, ]))


        #For base
        self.base_eps_net = base_model.diffusion.eps_net
        self.base_num_steps = num_steps
        self.base_trans_rot = base_model.diffusion.trans_rot
        self.base_trans_pos = base_model.diffusion.trans_pos
        self.base_trans_seq = base_model.diffusion.trans_seq

        self.register_buffer('base_position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('base_position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer('base__dummy', torch.empty([0, ]))



        #For aux
        self.aux_eps_net = aux_model.diffusion.eps_net
        self.aux_num_steps = num_steps
        self.aux_trans_rot = aux_model.diffusion.trans_rot
        self.aux_trans_pos = aux_model.diffusion.trans_pos
        self.aux_trans_seq = aux_model.diffusion.trans_seq

        self.register_buffer('aux_position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('aux_position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer('aux__dummy', torch.empty([0, ]))

        self.guidance_strength = guidance_strength

        self.guidance_strength_per_component = guidance_strength_per_component

        if self.guidance_strength_per_component:
            self.guidance_strength_pos = self.guidance_strength['guidance_strength_pos']
            self.guidance_strength_rot = self.guidance_strength['guidance_strength_rot']
            self.guidance_strength_seq = self.guidance_strength['guidance_strength_seq']

        self.config = config

    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p

    def forward(self, v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, denoise_structure, denoise_sequence, t=None):
        N, L = res_feat.shape[:2]
        if t == None:
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)
        p_0 = self._normalize_position(p_0)

        if denoise_structure:
            # Add noise to rotation
            R_0 = so3vec_to_rotation(v_0)
            v_noisy, _ = self.trans_rot.add_noise(v_0, mask_generate, t)
            # Add noise to positions
            p_noisy, eps_p = self.trans_pos.add_noise(p_0, mask_generate, t)
        else:
            R_0 = so3vec_to_rotation(v_0)
            v_noisy = v_0.clone()
            p_noisy = p_0.clone()
            eps_p = torch.zeros_like(p_noisy)

        if denoise_sequence:
            # Add noise to sequence
            _, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t)
        else:
            s_noisy = s_0.clone()

        beta = self.trans_pos.var_sched.betas[t]
        v_pred, R_pred, eps_p_pred, c_denoised = self.eps_net(
            v_noisy, p_noisy, s_noisy, res_feat, pair_feat, beta, mask_generate, mask_res
        )   # (N, L, 3), (N, L, 3, 3), (N, L, 3), (N, L, 20), (N, L)

        loss_dict = {}

        # Rotation loss
        loss_rot = rotation_matrix_cosine_loss(R_pred, R_0) # (N, L)
        loss_rot = (loss_rot * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['rot'] = loss_rot

        # Position loss
        loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction='none').sum(dim=-1)  # (N, L)
        loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['pos'] = loss_pos

        # Sequence categorical loss
        post_true = self.trans_seq.posterior(s_noisy, s_0, t)
        log_post_pred = torch.log(self.trans_seq.posterior(s_noisy, c_denoised, t) + 1e-8)
        kldiv = F.kl_div(
            input=log_post_pred, 
            target=post_true, 
            reduction='none',
            log_target=False
        ).sum(dim=-1)    # (N, L)
        loss_seq = (kldiv * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['seq'] = loss_seq

        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        v, p, s, 
        res_feat, pair_feat, 
        mask_generate, mask_res, 
        sample_structure=True, sample_sequence=True,
        pbar=False,
    ):
        """
        Args:
            v:  Orientations of contextual residues, (N, L, 3).
            p:  Positions of contextual residues, (N, L, 3).
            s:  Sequence of contextual residues, (N, L).
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            v_rand = random_uniform_so3([N, L], device=self._dummy.device)
            p_rand = torch.randn_like(p)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_rand, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_rand, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            s_rand = torch.randint_like(s, low=0, high=19)
            s_init = torch.where(mask_generate, s_rand, s)
        else:
            s_init = s

        traj = {self.num_steps: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x
        for t in pbar(range(self.num_steps, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj
    

    @torch.no_grad()
    def sample_for_sims(
        self, 
        
        #remove
        #v, p, s, 
        #res_feat, pair_feat, 
        #mask_generate, mask_res, 
        
        #base model
        base_v, base_p, base_s, 
        base_res_feat, base_pair_feat, 
        base_mask_generate, base_mask_res, 

        #aux model
        aux_v, aux_p, aux_s, 
        aux_res_feat, aux_pair_feat, 
        aux_mask_generate, aux_mask_res,  

        sample_structure=True, sample_sequence=True,
        pbar=False,

    
    ):
        """
        Args:
            v:  Orientations of contextual residues, (N, L, 3).
            p:  Positions of contextual residues, (N, L, 3).
            s:  Sequence of contextual residues, (N, L).
        """

        print('sampling for sims, diffusion, guidance strength:', self.guidance_strength)
        #N, L = v.shape[:2]
        #p = self._normalize_position(p)
        
        base_N, base_L = base_v.shape[:2]
        base_p = self._normalize_position(base_p)

        aux_N, aux_L = aux_v.shape[:2]
        aux_p = self._normalize_position(aux_p)
    

        # Set the orientation and position of residues to be predicted to random values -> maybe just using twice the base model
        # Randomizing and then using a learned guidance mask

        if sample_structure:
        
            #old
            #v_rand = random_uniform_so3([N, L], device=self._dummy.device)
            #p_rand = torch.randn_like(p)
            #v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_rand, v)
            #p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_rand, p)
            
            #check if moved to right device...
            base_v_rand = random_uniform_so3([base_N, base_L], device=self.base__dummy.device)
            base_p_rand = torch.randn_like(base_p)
            base_v_init = torch.where(base_mask_generate[:, :, None].expand_as(base_v), base_v_rand, base_v)
            base_p_init = torch.where(base_mask_generate[:, :, None].expand_as(base_p), base_p_rand, base_p)

            aux_v_rand = random_uniform_so3([aux_N, aux_L], device=self.aux__dummy.device)
            aux_p_rand = torch.randn_like(aux_p)
            aux_v_init = torch.where(aux_mask_generate[:, :, None].expand_as(aux_v), aux_v_rand, aux_v)
            aux_p_init = torch.where(aux_mask_generate[:, :, None].expand_as(aux_p), aux_p_rand, aux_p)
        
        else:
            raise ValueError("Sample structure must be True for sampling with sims")
            #v_init, p_init = v, p

        if sample_sequence:
            #old
            #s_rand = torch.randint_like(s, low=0, high=19)
            #s_init = torch.where(mask_generate, s_rand, s)
        
            #base
            base_s_rand = torch.randint_like(base_s, low=0, high=19)
            base_s_init = torch.where(base_mask_generate, base_s_rand, base_s)

            #aux
            aux_s_rand = torch.randint_like(aux_s, low=0, high=19)
            aux_s_init = torch.where(aux_mask_generate, aux_s_rand, aux_s)
        else:
            raise ValueError("Sample sequence must be True for sampling with sims")
            #s_init = s

        base_traj = {self.num_steps: (base_v_init, self._unnormalize_position(base_p_init), base_s_init)}
        
        #progress
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x


        #create the schedules
        omega_pos_schedule = None
        omega_rot_schedule = None
        omega_seq_schedule = None

        #first check if it exists
        if self.config.dgad.sims.guidance_schedule.enabled:
            print('using guidance schedule')
            omega_pos_schedule = get_schedule(self.guidance_strength_pos, self.config.dgad.sims.guidance_schedule.omega_pos_template)
            omega_rot_schedule = get_schedule(self.guidance_strength_rot, self.config.dgad.sims.guidance_schedule.omega_rot_template)
            omega_seq_schedule = get_schedule(self.guidance_strength_seq, self.config.dgad.sims.guidance_schedule.omega_seq_template)
        

        step_idx = 0
        for t in pbar(range(self.num_steps, 0, -1)):

            #old
            #v_t, p_t, s_t = traj[t]
            #p_t = self._normalize_position(p_t)
            
            #beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            #t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            #base
            base_v_t, base_p_t, base_s_t = base_traj[t]
            base_p_t = self._normalize_position(base_p_t)
            
            base_beta = self.base_trans_pos.var_sched.betas[t].expand([base_N, ])
            base_t_tensor = torch.full([base_N, ], fill_value=t, dtype=torch.long, device=self.base__dummy.device)


            #aux
            aux_v_t, aux_p_t, aux_s_t = base_traj[t]
            aux_p_t = self._normalize_position(aux_p_t)


            #aux_beta = self.aux_trans_pos.var_sched.betas[t].expand([aux_N, ])
            #Use same beta for aux
            aux_beta = base_beta

            #may need to be changed to be same as base
            aux_t_tensor = torch.full([aux_N, ], fill_value=t, dtype=torch.long, device=self.aux__dummy.device)



            # - - - - - - - - - - -

            # For the base model
            base_v_next, base_R_next, base_eps_p, base_c_denoised = self.base_eps_net(
                base_v_t, 
                base_p_t, 
                base_s_t, 
                base_res_feat, 
                base_pair_feat, 
                base_beta, 
                base_mask_generate, 
                base_mask_res,
                output_denoised_logits=self.config.dgad.sims.guidance_seq_on_logits
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)


            # For the base model
            aux_v_next, aux_R_next, aux_eps_p, aux_c_denoised = self.aux_eps_net(
                aux_v_t, 
                aux_p_t, 
                aux_s_t, 
                aux_res_feat,
                aux_pair_feat, 
                aux_beta,
                aux_mask_generate, 
                aux_mask_res,
                output_denoised_logits=self.config.dgad.sims.guidance_seq_on_logits
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)


            #
            #
            #

            # Anchor = current noisy state
            R_t = so3vec_to_rotation(base_v_t)   # (N, L, 3, 3)

            # Per-step decay (linear)
            T = self.num_steps
            
            if self.config.dgad.sims.decay:
                decay = (t - 1) / (T - 1)
            else:
                decay = 1


            if self.config.dgad.sims.guidance_schedule.enabled:
                omega_pos_t = omega_pos_schedule[step_idx]
                omega_rot_t = omega_rot_schedule[step_idx]
                omega_seq_t = omega_seq_schedule[step_idx]
            else:
                omega_pos_t = (getattr(self, "guidance_strength_pos", self.guidance_strength)) * decay
                omega_rot_t = (getattr(self, "guidance_strength_rot", self.guidance_strength)) * decay
                omega_seq_t = (getattr(self, "guidance_strength_seq", self.guidance_strength)) * decay

            print(f"t: {t} | gs_pos: {omega_pos_t} , gs_rot: {omega_rot_t}, gs_seq: {omega_seq_t}")

            # For the aux mode

            #do guidancen:
            #s_sims(xt, t) = s_base (xt, t) − guidance_strength(s_aux (xt, t) − sθr (xt, t)) = (1 + ω)sθr (xt, t) − ωsθs (xt, t).
            #(1 + ω)sθr (xt, t) − ωsθs (xt, t).
            
            #s_sims =  (1+guidance_strength)s_base(xt,t) - guidance_strength*s_aux(xt,t)

            # Guidance on position


    

            if self.config.dgad.sims.pos:
                print('guidance on position')
                sim_eps_p = ((1+omega_pos_t) * base_eps_p) - (omega_pos_t * aux_eps_p)
            else:
                sim_eps_p = base_eps_p
            
            # Guidance on rotation
            if self.config.dgad.sims.rot:
                print('guidance on rotation')
                # sims_v = lie_algebra_guidance_so3(
                #     base_v_next,
                #     aux_v_next,
                #     self.guidance_strength,
                # )
                # sims_v = lie_algebra_guidance_so3_anchored(
                #     base_v_next,
                #     aux_v_next,
                #     omega_rot_t,
                #     R_t,
                #     push_away=True,
                #     scale_match=True,
                # )

                sims_v = lie_algebra_guidance_so3_anchored_clipped(
                    base_v_next,
                    aux_v_next,
                    omega_rot_t,
                    R_t,
                )



            else:
                sims_v = base_v_next

            # Guidance on sequence
            if self.config.dgad.sims.seq:
                print('guidance on sequence')
                sims_c_logits = ((1 + omega_seq_t) * base_c_denoised) - (omega_seq_t * aux_c_denoised)
                #now do the softmax
                #sims_c = sims_c_logits
                #if self.config.dgad.sims.guidance_seq_on_logits:
                #always softmax for safety
                sims_c = F.softmax(sims_c_logits, dim=-1)

            else:
                sims_c = base_c_denoised
            
 
            v_next = self.base_trans_rot.denoise(base_v_t, sims_v, base_mask_generate, base_t_tensor)
            p_next = self.base_trans_pos.denoise(base_p_t, sim_eps_p, base_mask_generate, base_t_tensor)
            _, s_next = self.base_trans_seq.denoise(base_s_t, sims_c, base_mask_generate, base_t_tensor)

            
            
            if not sample_structure:
                raise ValueError("Sample structure must be True for sampling with sims")
                #v_next, p_next = v_t, p_t
            if not sample_sequence:
                raise ValueError("Sample structure must be True for sampling with sims")
                #s_next = s_t

            
            base_traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            base_traj[t] = tuple(x.cpu() for x in base_traj[t])    # Move previous states to cpu memory.
            step_idx += 1

        return base_traj
    



    @torch.no_grad()
    def sample_for_aux_model(
        self, 
        v, p, s, 
        res_feat, pair_feat, 
        mask_generate, mask_res, 
        sample_structure=True, sample_sequence=True,
        pbar=False,
    ):
        """
        Args:
            v:  Orientations of contextual residues, (N, L, 3).
            p:  Positions of contextual residues, (N, L, 3).
            s:  Sequence of contextual residues, (N, L).
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            v_rand = random_uniform_so3([N, L], device=self._dummy.device)
            p_rand = torch.randn_like(p)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_rand, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_rand, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            s_rand = torch.randint_like(s, low=0, high=19)
            s_init = torch.where(mask_generate, s_rand, s)
        else:
            s_init = s

        traj = {self.num_steps: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x
        for t in pbar(range(self.num_steps, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            #traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.
            #Throw away prev step don't need it anymore
            del traj[t]

        #Only return the last step 
        return traj[0]

    @torch.no_grad()
    def optimize(
        self, 
        v, p, s, 
        opt_step: int,
        res_feat, pair_feat, 
        mask_generate, mask_res, 
        sample_structure=True, sample_sequence=True,
        pbar=False,
    ):
        """
        Description:
            First adds noise to the given structure, then denoises it.
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)
        t = torch.full([N, ], fill_value=opt_step, dtype=torch.long, device=self._dummy.device)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            # Add noise to rotation
            v_noisy, _ = self.trans_rot.add_noise(v, mask_generate, t)
            # Add noise to positions
            p_noisy, _ = self.trans_pos.add_noise(p, mask_generate, t)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_noisy, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_noisy, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            _, s_noisy = self.trans_seq.add_noise(s, mask_generate, t)
            s_init = torch.where(mask_generate, s_noisy, s)
        else:
            s_init = s

        traj = {opt_step: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=opt_step, desc='Optimizing')
        else:
            pbar = lambda x: x
        for t in pbar(range(opt_step, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj






