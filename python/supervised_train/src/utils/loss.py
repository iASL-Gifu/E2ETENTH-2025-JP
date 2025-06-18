import torch

def heteroscedastic_loss(
    steer_mu_pred: torch.Tensor, 
    speed_mu_pred: torch.Tensor, 
    steer_log_sigma_squared_pred: torch.Tensor,
    speed_log_sigma_squared_pred: torch.Tensor,
    steer_true: torch.Tensor, 
    speed_true: torch.Tensor
) -> torch.Tensor:
    
    # 分散を計算 (exp を取ることで正の値に保証)
    steer_sigma_squared_pred = torch.exp(steer_log_sigma_squared_pred)
    speed_sigma_squared_pred = torch.exp(speed_log_sigma_squared_pred)
    
    # Steerの損失
    loss_steer = 0.5 * torch.mean(
        (steer_true - steer_mu_pred)**2 / steer_sigma_squared_pred + steer_log_sigma_squared_pred
    )
    
    # Speedの損失
    loss_speed = 0.5 * torch.mean(
        (speed_true - speed_mu_pred)**2 / speed_sigma_squared_pred + speed_log_sigma_squared_pred
    )
    
    total_loss = loss_steer + loss_speed
    return total_loss
