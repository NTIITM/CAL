"""
Contrastive Attention Learning (CAL) for scGPT.

Designed for discriminative classifiers.

Core idea: The CLS token's attention distribution over genes should be
cell-type-specific — same-type cells look at similar genes, different-type
cells look at different genes.

Loss = Supervised Contrastive on CLS-token attention distributions:
  - Same class pair (i,j): minimize JSD(a_i, a_j)  → pull together
  - Diff class pair (i,j): maximize JSD(a_i, a_j)  → push apart
  
Uses InfoNCE-style formulation for stability.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def compute_jsd_matrix(attns: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute pairwise JSD between all attention distributions in a batch.
    
    [CITATION] This implements the core component of the Jensen-Shannon
    Divergence cross-head representation heterogeneity measure (Eq 3 in the MH-CAL paper).

    
    Args:
        attns: (batch, seq_len) — CLS token's attention distribution per sample
    Returns:
        jsd_matrix: (batch, batch) pairwise JSD values
    """
    B = attns.shape[0]
    attns = attns.clamp(min=eps)
    # Normalize to valid distributions
    attns = attns / attns.sum(dim=-1, keepdim=True)
    
    # Expand for pairwise computation
    p = attns.unsqueeze(1).expand(B, B, -1)  # (B, B, seq)
    q = attns.unsqueeze(0).expand(B, B, -1)  # (B, B, seq)
    m = 0.5 * (p + q)
    
    kl_pm = (p * (p / m).log()).sum(dim=-1)  # (B, B)
    kl_qm = (q * (q / m).log()).sum(dim=-1)  # (B, B)
    jsd = 0.5 * (kl_pm + kl_qm)
    
    return jsd


class CALLoss(torch.nn.Module):
    """
    Contrastive Attention Learning Loss.
    
    Supervised contrastive loss on CLS token attention distributions.
    With Memory Queue support for very small batch sizes (e.g. batch_size=2).
    """
    
    def __init__(self, lambda_attn: float = 0.5, temperature: float = 0.1, memory_size: int = 128,
                 use_class_balanced_queue: bool = False, num_classes: int = 0, max_per_class: int = 16, **kwargs):
        super().__init__()
        self.lambda_attn = lambda_attn
        self.temperature = temperature
        self.memory_size = memory_size
        self.use_class_balanced_queue = use_class_balanced_queue
        self.num_classes = num_classes
        self.max_per_class = max_per_class
        
        if not self.use_class_balanced_queue:
            self.register_buffer("memory_attn", None)
            self.register_buffer("memory_labels", None)
            self.memory_ptr = 0
        else:
            self.register_buffer("cbq_attn", None)
            self.register_buffer("cbq_labels", None)
            self.register_buffer("cbq_ptr", torch.zeros(self.num_classes, dtype=torch.long))
            self.register_buffer("cbq_count", torch.zeros(self.num_classes, dtype=torch.long))
    
    def forward(
        self,
        current_attn: Tensor,
        labels: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if current_attn.dim() == 3:
            batch_size, seq_len, _ = current_attn.shape
            cls_attn = current_attn[:, 0, :]
        else:
            batch_size, seq_len = current_attn.shape
            cls_attn = current_attn
        
        if src_key_padding_mask is not None:
            valid = ~src_key_padding_mask[:, :seq_len]
            cls_attn = cls_attn * valid.float()
        
        cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 1. Update Memory Queue (detached)
        # [CITATION] This implements the dynamic memory queue mechanism (Eq 7 in the MH-CAL paper).
        with torch.no_grad():
            if self.use_class_balanced_queue:
                if self.cbq_attn is not None:
                    mem_seq_len = self.cbq_attn.shape[2]
                    cur_seq_len = cls_attn.shape[1]
                    if mem_seq_len < cur_seq_len:
                        pad_len = cur_seq_len - mem_seq_len
                        self.cbq_attn = torch.nn.functional.pad(self.cbq_attn, (0, pad_len), value=0.0)
                    elif cur_seq_len < mem_seq_len:
                        pad_len = mem_seq_len - cur_seq_len
                        cls_attn = torch.nn.functional.pad(cls_attn, (0, pad_len), value=0.0)
                else:
                    self.cbq_attn = torch.zeros(self.num_classes, self.max_per_class, cls_attn.shape[1], device=cls_attn.device)
                    self.cbq_labels = torch.zeros(self.num_classes, self.max_per_class, device=labels.device, dtype=labels.dtype) - 1
                
                detached_attn = cls_attn.detach()
                
                # Update per class
                for i in range(batch_size):
                    c = labels[i].item()
                    ptr = self.cbq_ptr[c].item()
                    self.cbq_attn[c, ptr] = detached_attn[i]
                    self.cbq_labels[c, ptr] = labels[i]
                    self.cbq_ptr[c] = (ptr + 1) % self.max_per_class
                    if self.cbq_count[c] < self.max_per_class:
                        self.cbq_count[c] += 1
                        
                total_valid = self.cbq_count.sum().item()
                if total_valid < 2:
                    return torch.tensor(0.0, device=current_attn.device, requires_grad=True)
                
                mem_attn_list = []
                mem_labels_list = []
                for c in range(self.num_classes):
                    cnt = self.cbq_count[c].item()
                    if cnt > 0:
                        mem_attn_list.append(self.cbq_attn[c, :cnt])
                        mem_labels_list.append(self.cbq_labels[c, :cnt])
                mem_attn = torch.cat(mem_attn_list, dim=0)
                mem_labels = torch.cat(mem_labels_list, dim=0)
                
            else:
                if self.memory_attn is not None:
                    mem_seq_len = self.memory_attn.shape[1]
                    cur_seq_len = cls_attn.shape[1]
                    if mem_seq_len < cur_seq_len:
                        pad_len = cur_seq_len - mem_seq_len
                        self.memory_attn = torch.nn.functional.pad(self.memory_attn, (0, pad_len), value=0.0)
                    elif cur_seq_len < mem_seq_len:
                        pad_len = mem_seq_len - cur_seq_len
                        cls_attn = torch.nn.functional.pad(cls_attn, (0, pad_len), value=0.0)
                else:
                    self.memory_attn = torch.zeros(self.memory_size, cls_attn.shape[1], device=current_attn.device)
                    self.memory_labels = torch.zeros(self.memory_size, device=labels.device, dtype=labels.dtype) - 1
                
                detached_attn = cls_attn.detach()
                
                # Enqueue
                if batch_size <= self.memory_size:
                    end_ptr = min(self.memory_ptr + batch_size, self.memory_size)
                    n = end_ptr - self.memory_ptr
                    self.memory_attn[self.memory_ptr:end_ptr] = detached_attn[:n]
                    self.memory_labels[self.memory_ptr:end_ptr] = labels[:n]
                    self.memory_ptr = (self.memory_ptr + n) % self.memory_size
                    
                valid_memory = (self.memory_labels != -1)
                if valid_memory.sum() < 2:
                    return torch.tensor(0.0, device=current_attn.device, requires_grad=True)
                    
                mem_attn = self.memory_attn[valid_memory]  # (M, seq_len)
                mem_labels = self.memory_labels[valid_memory]  # (M,)
                
        M = mem_attn.shape[0]
        
        # 2. Compute Match against Memory
        # [CITATION] Computes the InfoNCE-style Contrastive Attention Loss (Eq 8 in the MH-CAL paper).
        p = cls_attn.unsqueeze(1).expand(batch_size, M, -1)  # (B, M, S)
        q = mem_attn.unsqueeze(0).expand(batch_size, M, -1)  # (B, M, S)
        m = 0.5 * (p + q)
        
        p_clamp = p.clamp(min=1e-8)
        q_clamp = q.clamp(min=1e-8)
        m_clamp = m.clamp(min=1e-8)
        
        kl_pm = (p_clamp * (p_clamp / m_clamp).log()).sum(dim=-1)  # (B, M)
        kl_qm = (q_clamp * (q_clamp / m_clamp).log()).sum(dim=-1)  # (B, M)
        jsd_matrix = 0.5 * (kl_pm + kl_qm)  # (B, M)
        
        # Build label mask: same_label(i,j) = True if labels match
        labels_col = labels.unsqueeze(1)  # (B, 1)
        labels_row = mem_labels.unsqueeze(0)  # (1, M)
        same_label = (labels_col == labels_row)  # (B, M)
        
        n_positives = same_label.float().sum(dim=1)  # (B,)
        has_positive = n_positives > 0
        
        if has_positive.sum() == 0:
            return torch.tensor(0.0, device=current_attn.device, requires_grad=True)
            
        sim_matrix = -jsd_matrix / self.temperature  # (B, M)
        
        log_sum_all = torch.logsumexp(sim_matrix, dim=1)  # (B,)
        log_sum_pos = torch.logsumexp(sim_matrix.masked_fill(~same_label, -1e9), dim=1)  # (B,)
        
        loss_per_sample = -log_sum_pos + log_sum_all  # (B,)
        cal_loss = loss_per_sample[has_positive].mean()
        
        return self.lambda_attn * cal_loss


class MultiHeadCALLoss(torch.nn.Module):
    """
    Multi-Head Independent Contrastive Attention Learning Loss.
    
    Instead of averaging attention across heads (which dilutes discriminative signal),
    computes CAL loss independently per head then aggregates.
    
    Supports optional top-K head selection to focus on most discriminative heads.
    """
    
    def __init__(self, lambda_attn: float = 0.5, temperature: float = 0.1,
                 memory_size: int = 128, top_k_heads: int = 0, lambda_orth: float = 0.0):
        """
        Args:
            lambda_attn: weight for the per-head contrastive CAL loss
            temperature: temperature for InfoNCE
            memory_size: memory queue size per head
            top_k_heads: if > 0, only use the top-K most discriminative heads
            lambda_orth: weight for head orthogonality regularization (0 = disabled).
                         Adds L_{orth} = ||AA^T - I||_F^2 / H^2 where A is the
                         matrix of L2-normalized per-head mean attention profiles.
        """
        super().__init__()
        self.lambda_attn = lambda_attn
        self.temperature = temperature
        self.memory_size = memory_size
        self.top_k_heads = top_k_heads
        self.lambda_orth = lambda_orth
        
        # Lazily initialized per-head memory queues
        self.register_buffer("memory_attn", None)   # (H, memory_size, S)
        self.register_buffer("memory_labels", None)  # (memory_size,)
        self.memory_ptr = 0
        self.n_heads = None

    def _orth_loss(self, cls_attn: Tensor) -> Tensor:
        """
        Head orthogonality regularization.

        [CITATION] This implements the multi-head orthogonal regularization (Eq 9 in the MH-CAL paper).
        Forces different heads to attend to disjoint gene programs by
        penalizing the squared cosine similarity between any two heads'
        average attention profiles.

        Loss = ||A A^T - I||_F^2 / H^2

        where A ∈ R^{H×S} has L2-normalized rows (per-head mean attention).

        Args:
            cls_attn: (B, H, S) — per-head normalized CLS attention distributions
        Returns:
            Scalar orthogonality loss
        """
        # Batch-mean attention per head: (H, S)
        mean_attn = cls_attn.mean(dim=0)
        # L2-normalize each head's mean attention profile
        mean_attn = F.normalize(mean_attn, dim=-1, eps=1e-8)  # (H, S)
        # Gram matrix of head profiles: (H, H)
        gram = torch.mm(mean_attn, mean_attn.T)
        # Penalize deviation from identity: || gram - I ||_F^2 / H^2
        H = mean_attn.shape[0]
        I = torch.eye(H, device=mean_attn.device, dtype=mean_attn.dtype)
        loss = ((gram - I) ** 2).sum() / (H * H)
        return loss
    
    def _init_memory(self, n_heads: int, seq_len: int, device, label_device, label_dtype):
        self.n_heads = n_heads
        self.memory_attn = torch.zeros(n_heads, self.memory_size, seq_len, device=device)
        self.memory_labels = torch.zeros(self.memory_size, device=label_device, dtype=label_dtype) - 1
        self.memory_ptr = 0
    
    def forward(
        self,
        multi_head_attn: Tensor,
        labels: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            multi_head_attn: (B, H, S, S) — raw per-head attention weights
            labels: (B,)
            src_key_padding_mask: (B, S) — True for padding positions
        Returns:
            Scalar CAL loss
        """
        if multi_head_attn.ndim == 4:
            B, H, _, S = multi_head_attn.shape
            # Extract CLS row from each head: (B, H, S)
            cls_attn = multi_head_attn[:, :, 0, :]
        elif multi_head_attn.ndim == 3:
            B, H, S = multi_head_attn.shape
            cls_attn = multi_head_attn
        else:
            raise ValueError(f"Expected 3D or 4D attention tensor, got {multi_head_attn.ndim}D")
        
        # Mask padding positions
        if src_key_padding_mask is not None:
            valid = (~src_key_padding_mask).float().unsqueeze(1)  # (B, 1, S)
            cls_attn = cls_attn * valid
        
        # Normalize each head independently to valid distributions
        cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-8)  # (B, H, S)
        
        # Initialize or resize memory
        with torch.no_grad():
            if self.memory_attn is None:
                self._init_memory(H, S, cls_attn.device, labels.device, labels.dtype)
            else:
                mem_S = self.memory_attn.shape[2]
                if mem_S < S:
                    self.memory_attn = F.pad(self.memory_attn, (0, S - mem_S), value=0.0)
                elif S < mem_S:
                    cls_attn = F.pad(cls_attn, (0, mem_S - S), value=0.0)
                    S = mem_S
            
            # Enqueue current batch (detached)
            detached = cls_attn.detach()  # (B, H, S)
            end_ptr = min(self.memory_ptr + B, self.memory_size)
            n = end_ptr - self.memory_ptr
            # memory_attn is (H, memory_size, S), detached is (B, H, S) → permute to (H, B, S)
            self.memory_attn[:, self.memory_ptr:end_ptr, :] = detached[:n].permute(1, 0, 2)
            self.memory_labels[self.memory_ptr:end_ptr] = labels[:n]
            self.memory_ptr = (self.memory_ptr + n) % self.memory_size
            
            valid_mask = (self.memory_labels != -1)
            if valid_mask.sum() < 2:
                return torch.tensor(0.0, device=multi_head_attn.device, requires_grad=True)
            
            # Get valid memory entries
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            mem_labels = self.memory_labels[valid_indices]   # (M,)
            # mem_attn: (H, M, S)
            mem_attn = self.memory_attn[:, valid_indices, :]
        
        M = mem_labels.shape[0]
        
        # Build label mask (shared across heads): (B, M)
        same_label = (labels.unsqueeze(1) == mem_labels.unsqueeze(0))
        n_positives = same_label.float().sum(dim=1)
        has_positive = n_positives > 0
        
        if has_positive.sum() == 0:
            return torch.tensor(0.0, device=multi_head_attn.device, requires_grad=True)
        
        # Compute per-head JSD and contrastive loss
        head_losses = []
        for h in range(H):
            # cls_attn[:, h, :] → (B, S), mem_attn[h] → (M, S)
            p = cls_attn[:, h, :].unsqueeze(1).expand(B, M, -1)  # (B, M, S)
            q = mem_attn[h].unsqueeze(0).expand(B, M, -1)         # (B, M, S)
            m_dist = 0.5 * (p + q)
            
            p_c = p.clamp(min=1e-8)
            q_c = q.clamp(min=1e-8)
            m_c = m_dist.clamp(min=1e-8)
            
            kl_pm = (p_c * (p_c / m_c).log()).sum(dim=-1)  # (B, M)
            kl_qm = (q_c * (q_c / m_c).log()).sum(dim=-1)  # (B, M)
            jsd = 0.5 * (kl_pm + kl_qm)
            
            sim = -jsd / self.temperature  # (B, M)
            log_sum_all = torch.logsumexp(sim, dim=1)
            log_sum_pos = torch.logsumexp(sim.masked_fill(~same_label, -1e9), dim=1)
            
            loss_h = (-log_sum_pos + log_sum_all)  # (B,)
            loss_h = loss_h[has_positive].mean()
            head_losses.append(loss_h)
        
        head_losses = torch.stack(head_losses)  # (H,)
        
        if self.top_k_heads > 0 and self.top_k_heads < H:
            topk_losses, _ = torch.topk(head_losses, self.top_k_heads)
            total_loss = topk_losses.mean()
        else:
            total_loss = head_losses.mean()
        
        contrastive_loss = self.lambda_attn * total_loss

        # Head orthogonality regularization
        if self.lambda_orth > 0.0:
            orth_loss = self.lambda_orth * self._orth_loss(cls_attn)
        else:
            orth_loss = torch.tensor(0.0, device=multi_head_attn.device)

        # [CITATION] Total objective combining contrastive loss and orthogonality (Eq 10 in the MH-CAL paper).
        return contrastive_loss + orth_loss