import torch
import torch.nn as nn


class Dynamics(nn.Module):
    """Multi-step EEF dynamics model (Ctrl-World style MLP adapter).

    Input:
        - current_pose: [B, pose_dim]
        - action_seq: [B, T, action_dim]

    Output:
        - pred_pose_seq: [B, T, pose_dim]

    This model predicts *future absolute* EEF pose channels from the current pose and
    a whole action chunk, with one predicted future pose per input action.
    For LIBERO integration we use:
        action_dim = 6  (delta xyz + delta axis-angle)
        pose_dim   = 6  (absolute xyz + absolute axis-angle)

    Gripper is handled outside the model in rollout code.
    """

    def __init__(
        self,
        pose_dim: int = 6,
        action_dim: int = 6,
        action_num: int = 15,
        hidden_size: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        if action_num <= 0:
            raise ValueError("action_num must be positive")

        self.pose_dim = int(pose_dim)
        self.action_dim = int(action_dim)
        self.action_num = int(action_num)

        in_dim = self.pose_dim + self.action_num * self.action_dim
        out_dim = self.action_num * self.pose_dim
        layers = [nn.Linear(in_dim, hidden_size), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.SiLU()])
        layers.append(nn.Linear(hidden_size, out_dim))
        self.net = nn.Sequential(*layers)

        # Normalization buffers (broadcast shape: [1, 1, D]).
        self.register_buffer("action_min", torch.zeros(1, 1, self.action_dim, dtype=torch.float32), persistent=True)
        self.register_buffer("action_max", torch.ones(1, 1, self.action_dim, dtype=torch.float32), persistent=True)
        self.register_buffer("pose_min", torch.zeros(1, 1, self.pose_dim, dtype=torch.float32), persistent=True)
        self.register_buffer("pose_max", torch.ones(1, 1, self.pose_dim, dtype=torch.float32), persistent=True)
        self.register_buffer("has_stats", torch.tensor(False, dtype=torch.bool), persistent=True)

    @staticmethod
    def _to_tensor_stats(values, expected_dim: int) -> torch.Tensor:
        t = torch.as_tensor(values, dtype=torch.float32)
        if t.ndim == 2 and t.shape[0] == 1:
            t = t[0]
        if t.ndim != 1 or t.shape[0] != expected_dim:
            raise ValueError(f"Expected stats shape [{expected_dim}], got {tuple(t.shape)}")
        return t.view(1, 1, expected_dim)

    def set_normalization_stats(
        self,
        *,
        action_min,
        action_max,
        pose_min,
        pose_max,
    ) -> None:
        """Set min/max bounds used for [-1, 1] normalization."""
        self.action_min.copy_(self._to_tensor_stats(action_min, self.action_dim).to(self.action_min.device))
        self.action_max.copy_(self._to_tensor_stats(action_max, self.action_dim).to(self.action_max.device))
        self.pose_min.copy_(self._to_tensor_stats(pose_min, self.pose_dim).to(self.pose_min.device))
        self.pose_max.copy_(self._to_tensor_stats(pose_max, self.pose_dim).to(self.pose_max.device))
        self.has_stats.fill_(True)

    @staticmethod
    def _normalize_bound_torch(
        data: torch.Tensor,
        data_min: torch.Tensor,
        data_max: torch.Tensor,
        *,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        out = 2.0 * (data - data_min) / (data_max - data_min + eps) - 1.0
        return out.clamp(min=clip_min, max=clip_max)

    @staticmethod
    def _denormalize_bound_torch(
        data: torch.Tensor,
        data_min: torch.Tensor,
        data_max: torch.Tensor,
        *,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        scale = clip_max - clip_min + eps
        return (data - clip_min) / scale * (data_max - data_min) + data_min

    def normalize_action(self, action_seq: torch.Tensor) -> torch.Tensor:
        if not bool(self.has_stats.item()):
            raise RuntimeError("Normalization stats are not set")
        return self._normalize_bound_torch(action_seq, self.action_min, self.action_max)

    def normalize_pose(self, pose_seq: torch.Tensor) -> torch.Tensor:
        if not bool(self.has_stats.item()):
            raise RuntimeError("Normalization stats are not set")
        return self._normalize_bound_torch(pose_seq, self.pose_min, self.pose_max)

    def denormalize_pose(self, pose_seq_norm: torch.Tensor) -> torch.Tensor:
        if not bool(self.has_stats.item()):
            raise RuntimeError("Normalization stats are not set")
        return self._denormalize_bound_torch(pose_seq_norm, self.pose_min, self.pose_max)

    def forward(
        self,
        current_pose: torch.Tensor,
        action_seq: torch.Tensor,
        *,
        normalize_pose_input: bool = True,
        normalize_input: bool = True,
        denormalize_output: bool = False,
        strict_horizon: bool = True,
    ) -> torch.Tensor:
        """Predict absolute pose sequence from action sequence.

        Args:
            current_pose: [B, pose_dim]
            action_seq: [B, T, action_dim]
            normalize_pose_input: whether to normalize current pose by loaded stats.
            normalize_input: whether to normalize action by loaded stats.
            denormalize_output: whether to convert network output back to raw pose scale.
            strict_horizon: if True, require T == action_num.

        Returns:
            pred_pose_seq: [B, T, pose_dim]
        """
        if current_pose.ndim != 2:
            raise ValueError(f"current_pose must be [B, pose_dim], got {tuple(current_pose.shape)}")
        if current_pose.shape[-1] != self.pose_dim:
            raise ValueError(
                f"current_pose last dim must be pose_dim={self.pose_dim}, got {current_pose.shape[-1]}"
            )
        if action_seq.ndim != 3:
            raise ValueError(f"action_seq must be [B, T, action_dim], got {tuple(action_seq.shape)}")
        if action_seq.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_seq last dim must be action_dim={self.action_dim}, got {action_seq.shape[-1]}"
            )

        bsz, horizon, _ = action_seq.shape
        if strict_horizon and horizon != self.action_num:
            raise ValueError(f"Expected horizon={self.action_num}, got {horizon}")

        if current_pose.shape[0] != bsz:
            raise ValueError(
                f"current_pose batch size must match action_seq batch size, got {current_pose.shape[0]} vs {bsz}"
            )

        if normalize_pose_input:
            current_pose = self.normalize_pose(current_pose.unsqueeze(1)).squeeze(1)

        if normalize_input:
            action_seq = self.normalize_action(action_seq)

        if horizon != self.action_num:
            # For non-strict use-cases, pad/truncate to model horizon, then slice back.
            if horizon < self.action_num:
                pad = action_seq[:, -1:, :].expand(bsz, self.action_num - horizon, self.action_dim)
                action_seq_in = torch.cat([action_seq, pad], dim=1)
            else:
                action_seq_in = action_seq[:, : self.action_num, :]
        else:
            action_seq_in = action_seq

        flat = action_seq_in.reshape(bsz, self.action_num * self.action_dim)
        flat = torch.cat([current_pose, flat], dim=1)
        pred_flat = self.net(flat)
        pred = pred_flat.reshape(bsz, self.action_num, self.pose_dim)

        if horizon != self.action_num:
            pred = pred[:, :horizon, :]

        if denormalize_output:
            pred = self.denormalize_pose(pred)
        return pred

    def predict(self, current_pose: torch.Tensor, action_seq: torch.Tensor, *, strict_horizon: bool = True) -> torch.Tensor:
        """Inference helper: normalize input and denormalize output automatically."""
        with torch.no_grad():
            return self.forward(
                current_pose,
                action_seq,
                normalize_pose_input=True,
                normalize_input=True,
                denormalize_output=True,
                strict_horizon=strict_horizon,
            )
