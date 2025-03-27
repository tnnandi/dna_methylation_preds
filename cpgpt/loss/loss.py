import torch
import torch.nn.functional as F


def wd_loss(p_hat: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Calculates the Wasserstein distance (WD) loss between two probability distributions.

    Args:
        p_hat (torch.Tensor): Predicted probability distribution.
        p (torch.Tensor): Target probability distribution.

    Returns:
        torch.Tensor: The WD loss between the predicted and target distributions.

    """
    p_hat_sorted, _ = torch.sort(p_hat, dim=-1)
    p_sorted, _ = torch.sort(p, dim=-1)
    wd = torch.mean(torch.abs(p_hat_sorted - p_sorted), dim=-1)
    return wd.mean()


def kld_bernoulli_loss(p_hat: torch.Tensor, p: torch.Tensor, eps: float = 10e-7) -> torch.Tensor:
    """Computes the Kullback-Leibler Divergence (KLD) loss between two probability distributions.

    Args:
        p_hat (torch.Tensor): The predicted probability distribution.
        p (torch.Tensor): The target probability distribution.
        eps (float, optional): A small value added to the probabilities to avoid division
            by zero. Defaults to 10e-7.

    Returns:
        torch.Tensor: The mean KLD loss between the predicted and target distributions.

    """
    p_hat = p_hat.to(torch.float32)
    p = p.to(torch.float32)
    p_hat = torch.clip(p_hat, eps, 1 - eps)
    p = torch.clip(p, eps, 1 - eps)
    return p * torch.log(p / p_hat) + (1 - p) * torch.log((1 - p) / (1 - p_hat))


def beta_loss(p_hat: torch.Tensor, p: torch.Tensor, eps: float = 10e-7) -> torch.Tensor:
    """Calculates the Beta loss between two probability distributions.

    Args:
        p_hat (torch.Tensor): Predicted probability distribution.
        p (torch.Tensor): Target probability distribution.
        eps (float, optional): A small value added to the probabilities to avoid division
            by zero. Defaults to 10e-7.

    Returns:
        torch.Tensor: The Beta loss between the predicted and target distributions.

    """
    p_hat = torch.clip(p_hat, eps, 1 - eps)
    p = torch.clip(p, eps, 1 - eps)
    return -torch.log(1 - torch.abs(p - p_hat) + eps)


def contrastive_loss(cell_emb: torch.Tensor, ecs_threshold: float = 0.6) -> torch.Tensor:
    """Calculates the contrastive loss between cell embeddings.

    Args:
        cell_emb (torch.Tensor): The cell embeddings, with shape (batch_size, embedding_dim).
        ecs_threshold (float, optional): The threshold value for the contrastive loss.
            Defaults to 0.5.

    Returns:
        torch.Tensor: The contrastive loss value.

    """
    cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
    cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())
    mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
    cos_sim = cos_sim.masked_fill(mask, 0.0)
    cos_sim = F.relu(cos_sim)
    return torch.mean(1 - (cos_sim - ecs_threshold) ** 2)


def kld_normal_loss(sample_embedding: torch.Tensor) -> torch.Tensor:
    """Calculate KL divergence loss between the distribution of sample embeddings and N(0, 1).

    This function computes the Kullback-Leibler divergence between the distribution
    of the sample embeddings and a standard normal distribution N(0, 1).

    Args:
        sample_embedding (torch.Tensor): The sample embeddings, shape (batch_size, embedding_dim).

    Returns:
        torch.Tensor: The mean KL divergence loss across the batch.

    """
    # Compute mean and variance for each sample in the batch
    mu = torch.mean(sample_embedding, dim=1)
    var = torch.var(sample_embedding, dim=1, unbiased=False)

    # Clamp variance to avoid numerical issues
    var = torch.clamp(var, min=1e-4)

    # Calculate log variance
    log_var = torch.log(var)

    # KL divergence between N(mu, var) and N(0, 1)
    return 0.5 * torch.mean(mu.pow(2) + var - log_var - 1)


def gompertz_aft_loss(
    beta_x: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    lambda_param: torch.Tensor | float,
    gamma: torch.Tensor | float,
    epsilon: float = 1e-6,
    regularization_weight: float = 0.1,
    adaptive_regularization: bool = True,
) -> torch.Tensor:
    """Calculate the Gompertz Accelerated Failure Time (AFT) loss with learnable parameters.

    This improved implementation supports batch-wise or sample-wise lambda and gamma parameters,
    with adaptive regularization based on the current batch statistics.

    Args:
        beta_x (torch.Tensor): The linear predictor (beta * X) for each sample.
        time (torch.Tensor): The observed follow-up time for each sample.
        event (torch.Tensor): Binary indicator of whether the event occurred (1) or censored (0).
        lambda_param (torch.Tensor | float): The scale parameter(s) of the Gompertz distribution.
            Can be a scalar float or a tensor for batch/sample-specific values.
        gamma (torch.Tensor | float): The shape parameter(s) of the Gompertz distribution.
            Can be a scalar float or a tensor for batch/sample-specific values.
        epsilon (float, optional): Small constant to avoid numerical instability. Defaults to 1e-6.
        regularization_weight (float, optional): Weight for parameter regularization. 
            Defaults to 0.1.
        adaptive_regularization (bool, optional): Whether to use adaptive regularization 
            based on batch statistics. Defaults to True.

    Returns:
        torch.Tensor: The negative log-likelihood loss with parameter regularization.
    """
    # Convert parameters to tensors if they're scalars
    if isinstance(lambda_param, float):
        lambda_param = torch.tensor(lambda_param, device=beta_x.device)
    if isinstance(gamma, float):
        gamma = torch.tensor(gamma, device=beta_x.device)
    
    # Ensure parameters have proper shape for broadcasting
    batch_size = beta_x.size(0)
    if lambda_param.dim() == 0:
        lambda_param = lambda_param.expand(batch_size)
    if gamma.dim() == 0:
        gamma = gamma.expand(batch_size)
    
    # Ensure positive values for lambda and gamma (via softplus or clamp)
    lambda_param = torch.nn.functional.softplus(lambda_param)
    gamma = torch.nn.functional.softplus(gamma)
    
    # Apply reasonable bounds to prevent extreme values
    lambda_param = torch.clamp(lambda_param, min=1e-4, max=0.05)
    gamma = torch.clamp(gamma, min=0.01, max=0.3)
    
    # Compute the linear predictor term
    exp_beta_x = torch.exp(beta_x)
    
    # Compute the time-dependent term for the Gompertz model (with stability measures)
    # Reshape parameters for proper broadcasting
    lambda_param_view = lambda_param.view(-1, 1) if lambda_param.dim() == 1 else lambda_param
    gamma_view = gamma.view(-1, 1) if gamma.dim() == 1 else gamma
    
    # Gompertz AFT survival function
    exp_term = torch.exp(gamma_view * time * exp_beta_x)
    # Clip exponentiated term to prevent overflow
    exp_term = torch.clamp(exp_term, min=1.0 + epsilon, max=50.0)
    
    # Compute survival function (S) with numerical stability
    log_S = -(lambda_param_view / gamma_view) * (exp_term - 1)
    S = torch.exp(torch.clamp(log_S, max=0.0))
    
    # Compute hazard function (h) with numerical stability
    h = lambda_param_view * exp_term
    h = torch.clamp(h, min=epsilon)
    
    # Log-likelihood for uncensored data (event occurred)
    ll_uncensored = event * (torch.log(h) + log_S)
    
    # Log-likelihood for censored data
    ll_censored = (1 - event) * torch.log(S)
    
    # Total negative log-likelihood (per sample)
    nll = -(ll_uncensored + ll_censored)
    
    # Apply adaptive regularization if enabled
    if adaptive_regularization:
        # Calculate median values for this batch to use as regularization targets
        lambda_median = lambda_param.median()
        gamma_median = gamma.median()
        
        # Compute parameter variance within the batch
        lambda_var = ((lambda_param - lambda_median) ** 2).mean()
        gamma_var = ((gamma - gamma_median) ** 2).mean()
        
        # Add regularization to promote consistency across the batch
        # This helps stabilize learning when batch size is small
        reg_loss = regularization_weight * (lambda_var + gamma_var)
        
        # Add a weak L2 penalty to keep parameters close to reasonable values
        lambda_center = 0.005  # Center point for lambda regularization
        gamma_center = 0.1     # Center point for gamma regularization
        center_reg = 0.01 * ((lambda_median - lambda_center)**2 + (gamma_median - gamma_center)**2)
        
        return nll.mean() + reg_loss + center_reg
    else:
        # Traditional fixed-range regularization
        lambda_range = (0.0001, 0.01)
        gamma_range = (0.05, 0.15)
        
        # Calculate per-parameter penalties
        lambda_penalty = torch.mean(
            (torch.relu(lambda_param - lambda_range[1]) + 
             torch.relu(lambda_range[0] - lambda_param)) ** 2
        )
        gamma_penalty = torch.mean(
            (torch.relu(gamma - gamma_range[1]) + 
             torch.relu(gamma_range[0] - gamma)) ** 2
        )
        
        # Add penalties with weight
        return nll.mean() + regularization_weight * (lambda_penalty + gamma_penalty)


def cph_loss(pred: torch.Tensor, times: torch.Tensor, events: torch.Tensor, 
             l2_reg: float = 0.0, ties_method: str = 'efron') -> torch.Tensor:
    """Computes an improved Cox Proportional Hazards loss function for a batch.

    Args:
        pred (torch.Tensor): Predicted risk scores from the neural network (shape: [batch_size]).
        times (torch.Tensor): Survival times or time to censoring (shape: [batch_size]).
        events (torch.Tensor): Event indicators (1 if the event occurred, 0 if censored)
            (shape: [batch_size]).
        l2_reg (float, optional): L2 regularization strength. Defaults to 0.0.
        ties_method (str, optional): Method for handling ties in survival times.
            Options are 'breslow' or 'efron'. Defaults to 'efron'.

    Returns:
        torch.Tensor: The computed Cox loss for the batch.
    """
    # Ensure tensors are of the same shape
    pred = pred.view(-1)
    times = times.view(-1)
    events = events.view(-1)
    
    # Check for empty batch or no events
    if pred.size(0) == 0 or torch.sum(events) == 0:
        return torch.tensor(0.0, device=pred.device)

    # Sort by descending survival times
    order = torch.argsort(times, descending=True)
    pred = pred[order]
    times = times[order]
    events = events[order]
    
    # Compute risk scores
    risk_scores = pred
    exp_risk = torch.exp(risk_scores)
    
    # Handle ties using the specified method
    if ties_method == 'breslow':
        # Breslow method (current implementation)
        # Compute the log cumulative sum of the exponentials of the predictions
        log_cum_sum = torch.logcumsumexp(risk_scores, dim=0)
        
        # Select the events (uncensored data)
        event_mask = events == 1
        event_indices = torch.nonzero(event_mask).squeeze()
        
        # Handle cases where there might be only one event in the batch
        if event_indices.ndim == 0:
            event_indices = event_indices.unsqueeze(0)
        
        # Get the predictions and log cumulative sums for the events
        risk_scores_events = risk_scores[event_indices]
        log_cum_sum_events = log_cum_sum[event_indices]
        
        # Compute the Cox loss
        loss = -torch.sum(risk_scores_events - log_cum_sum_events)
    
    elif ties_method == 'efron':
        # Efron method for handling ties in survival times
        # Group by unique times
        unique_times, inverse_indices = torch.unique(times, sorted=True, return_inverse=True)
        unique_times = unique_times.flip(0)  # Descending order
        
        loss = torch.tensor(0.0, device=pred.device)
        cum_exp_risk = torch.zeros_like(exp_risk)
        
        # Process each unique time point
        for i, t in enumerate(unique_times):
            at_risk = times >= t
            risk_set = exp_risk[at_risk]
            cum_exp_risk = torch.sum(risk_set)
            
            # Get events at this time point
            events_at_t = (times == t) & (events == 1)
            sum_events_at_t = events_at_t.sum()
            
            if sum_events_at_t > 0:
                # Get risk scores for events at this time
                risk_scores_at_t = risk_scores[events_at_t]
                exp_risk_at_t = exp_risk[events_at_t]
                sum_exp_risk_at_t = exp_risk_at_t.sum()
                
                # Efron's method for handling ties
                for j in range(sum_events_at_t):
                    frac = j / sum_events_at_t
                    loss -= torch.sum(risk_scores_at_t) / sum_events_at_t - \
                           torch.log(cum_exp_risk - frac * sum_exp_risk_at_t)
    else:
        raise ValueError(f"Unknown ties method: {ties_method}. Use 'breslow' or 'efron'.")
    
    # Add L2 regularization
    if l2_reg > 0:
        l2_penalty = l2_reg * torch.sum(pred ** 2) / pred.size(0)
        loss += l2_penalty
    
    # Normalize the loss
    num_events = events.sum()
    loss = loss / num_events if num_events > 0 else loss
    
    # Return mean loss per sample
    batch_size = pred.size(0)
    return loss / batch_size


def c_index_loss(
    predicted_risks: torch.Tensor, 
    actual_times: torch.Tensor, 
    events: torch.Tensor,
    age: torch.Tensor,
    sex: torch.Tensor,
    sigma: float = 0.01,
    time_weights: bool = True,
    beta: float = 0.1,
    gompertz_transform: bool = False,
    baseline_weights: tuple = (0.025, -0.15)  # Hyperparameters for age and sex based on GrimAge2
) -> torch.Tensor:
    """
    Compute a differentiable approximation of the C-index loss that captures
    the risk prediction additional to age and sex.
    
    This function computes a baseline risk using a linear combination of age and sex:
        baseline_risk = baseline_weights[0] * age + baseline_weights[1] * sex
    and then calculates the "additional" risk as:
        additional_risk = predicted_risks - baseline_risk
    The loss is then computed on these residuals.
    
    Args:
        predicted_risks (torch.Tensor): Predicted risk scores (higher implies higher risk).
        actual_times (torch.Tensor): Observed times (for events or censoring).
        events (torch.Tensor): Event indicators (1 if event occurred, 0 if censored).
        age (torch.Tensor): Age labels (continuous).
        sex (torch.Tensor): Sex labels (e.g., 0 or 1).
        sigma (float, optional): Smoothing parameter for the sigmoid function. Defaults to 0.1.
        time_weights (bool, optional): Whether to weight pairs by time differences. Defaults to True.
        beta (float, optional): L2 regularization strength on additional risks. Defaults to 0.001.
        gompertz_transform (bool, optional): Whether to apply a gompertz transform to the predicted risks. Defaults to True.
        baseline_weights (tuple, optional): Coefficients for age and sex in the baseline risk model.
                                          Defaults to (0.01, 0.01).
    
    Returns:
        torch.Tensor: Adjusted c-index loss (to be minimized).
    """
    # Compute baseline risk from age and sex
    baseline_risk = baseline_weights[0] * age.float() + baseline_weights[1] * sex.float()

    # Apply gompertz transform if requested
    if gompertz_transform:
        predicted_risks = torch.exp(predicted_risks) * torch.exp(0.1 * predicted_risks)
    
    # Compute the additional risk (i.e. risk beyond age and sex)
    additional_risk = predicted_risks.float().squeeze(1) - baseline_risk
    
    # Flatten inputs for pairwise computations
    additional_risk = additional_risk.view(-1)
    actual_times = actual_times.float().view(-1)
    events = events.float().view(-1)
    
    n = additional_risk.size(0)
    if n == 0:
        return torch.tensor(0.0, device=predicted_risks.device)
    
    # Expand dimensions for pairwise comparisons
    risk_i = additional_risk.unsqueeze(0).expand(n, n)
    risk_j = additional_risk.unsqueeze(1).expand(n, n)
    time_i = actual_times.unsqueeze(0).expand(n, n)
    time_j = actual_times.unsqueeze(1).expand(n, n)
    event_i = events.unsqueeze(0).expand(n, n)
    event_j = events.unsqueeze(1).expand(n, n)
    
    # Define valid pairs:
    # - subject i had an event and occurred before subject j, or
    # - tie in time where subject i had event and subject j was censored.
    valid = ((time_i < time_j) & (event_i == 1)) | (
             (time_i == time_j) & (event_i == 1) & (event_j == 0))
    
    # Exclude self-comparisons
    valid = valid & (~torch.eye(n, dtype=torch.bool, device=predicted_risks.device))
    
    if not torch.any(valid):
        return torch.tensor(0.0, device=predicted_risks.device)
    
    # Compute weights (e.g., time-dependent weighting)
    weights = torch.ones_like(valid, dtype=torch.float)
    if time_weights:
        time_diff = torch.abs(time_i - time_j)
        weights = torch.log1p(time_diff) / torch.log(torch.tensor(2.0))
        weights = torch.clamp(weights, min=0.1, max=10.0)
    
    weights = weights * valid.float()
    
    # Compute pairwise risk differences
    risk_diff = risk_i - risk_j
    
    # To handle ties in predicted risks, add a small random noise
    tie_noise = (torch.rand_like(risk_diff) - 0.5) * 1e-6
    risk_diff = risk_diff + tie_noise * (torch.abs(risk_diff) < 1e-6).float()
    
    # Smooth approximation using sigmoid
    indicator = torch.sigmoid(risk_diff / sigma)
    
    # Compute the weighted concordance
    concordance = torch.sum(weights * indicator) / (torch.sum(weights) + 1e-8)
    
    # Loss is 1 - concordance
    loss = 1 - concordance
    
    # Optionally add L2 regularization on the additional risks
    if beta > 0:
        l2_reg = beta * torch.mean(additional_risk ** 2)
        loss = loss + l2_reg
    
    return loss


def robust_loss(
    cls_embedding_current: torch.Tensor,
    cls_embedding_previous: torch.Tensor,
    temperature: float = 0.15,
) -> torch.Tensor:
    """Compute improved robust loss using an adapted InfoNCE approach for two CLS embeddings.

    Args:
        cls_embedding_current (torch.Tensor): Current step's CLS embeddings of shape
            (batch_size, embedding_dim)
        cls_embedding_previous (torch.Tensor): Previous step's CLS embeddings of shape
            (batch_size, embedding_dim)
        temperature (float): Temperature parameter to scale the similarities

    Returns:
    torch.Tensor: Scalar loss value

    """
    batch_size = cls_embedding_current.shape[0]

    # Normalize embeddings
    cls_current_norm = F.normalize(cls_embedding_current, p=2, dim=1)
    cls_previous_norm = F.normalize(cls_embedding_previous, p=2, dim=1)

    # Compute similarity matrix
    sim_matrix = torch.matmul(cls_current_norm, cls_previous_norm.T) / temperature

    # Positive similarities are on the diagonal (same sample, different steps)
    positive_sim = sim_matrix.diag().unsqueeze(1)

    # All similarities are used for normalization (excluding the diagonal in log_sum_exp)
    mask = torch.eye(batch_size, dtype=torch.bool, device=cls_embedding_current.device)
    exp_sim_matrix = torch.exp(sim_matrix)
    exp_sim_matrix = exp_sim_matrix.masked_fill(mask, 0)

    # Compute log-sum-exp
    log_sum_exp = torch.log(exp_sim_matrix.sum(dim=1, keepdim=True) + 1e-7)

    # Compute loss
    loss = -positive_sim + log_sum_exp

    return loss.mean()


def censored_mae_loss(
    predicted_times: torch.Tensor,
    actual_times: torch.Tensor,
    events: torch.Tensor,
    uncensored_weight: float = 2.0,
) -> torch.Tensor:
    """Censored Mean Absolute Error Loss for time-to-event prediction.

    Args:
        predicted_times (torch.Tensor): Predicted times to the event.
        actual_times (torch.Tensor): Observed times (event or censoring).
        events (torch.Tensor): Event indicators (1 if the event occurred, 0 if censored).
        uncensored_weight (float, optional): Weight for uncensored data. Defaults to 2.0.

    Returns:
        torch.Tensor: Mean Censored MAE loss value.

    """
    # MAE for uncensored data (event occurred)
    event_loss = uncensored_weight * events * torch.abs(predicted_times - actual_times)

    # Asymmetric MAE for censored data (penalize only if predicted < actual)
    censored_loss = (1 - events) * torch.max(
        torch.zeros_like(actual_times),
        actual_times - predicted_times,
    )

    # Combine losses
    total_loss = event_loss + censored_loss

    # Return mean loss with regularization
    return total_loss.mean()


def variance_regularization(
    z: torch.Tensor, eps: float = 1e-4, target_std: float = 1.0
) -> torch.Tensor:
    """Regularizes the variance of the embedding features across the batch.

    Penalizing dimensions whose standard deviation falls below a given threshold.

    Args:
        z (torch.Tensor): Embeddings of shape (batch_size, embedding_dim).
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-4.
        target_std (float, optional): The minimally acceptable standard deviation.
            Defaults to 1.0.

    Returns:
        torch.Tensor: A scalar penalty that increases when the feature variance is too low.

    """
    # Compute standard deviation along the batch (for each feature dimension)
    std = torch.sqrt(torch.var(z, dim=0) + eps)
    # Penalize if the std is lower than the target standard deviation
    return F.relu(target_std - std).mean()


def consistency_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.01, variance_weight: float = 0.1
) -> torch.Tensor:
    """Compute an improved consistency loss using InfoNCE formulation.

    Uses the standard InfoNCE formulation as in our SimCE-like loss, but adds a variance
    regularization term to help prevent collapse in the embedding space.

    Steps:
      1. L2-normalize the two sets of embeddings.
      2. Concatenate them and compute the cosine similarity matrix (scaled by temperature).
      3. Mask self-similarities (using a safe minimum value for the current dtype).
      4. Construct target indices such that each sample's augmented pair is the positive.
      5. Compute the cross-entropy loss.
      6. Compute a variance regularization penalty for each set of normalized embeddings.
      7. Return the combined loss.

    Args:
        z1 (torch.Tensor): First set of embeddings, shape (batch_size, embedding_dim).
        z2 (torch.Tensor): Second set of embeddings, shape (batch_size, embedding_dim).
        temperature (float, optional): Temperature for scaling similarity scores.
            Defaults to 0.05.
        variance_weight (float, optional): Scaling factor for the variance regularization term.
            Defaults to 0.1.

    Returns:
        torch.Tensor: The combined loss value.

    """
    batch_size = z1.shape[0]

    # L2-normalize the embeddings.
    z1_norm = F.normalize(z1, p=2, dim=-1)
    z2_norm = F.normalize(z2, p=2, dim=-1)

    # Concatenate the normalized embeddings.
    embeddings = torch.cat([z1_norm, z2_norm], dim=0)

    # Compute the cosine similarity matrix (scaled by temperature).
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # Mask out self-similarities using a safe minimum value.
    mask = torch.eye(2 * batch_size, device=embeddings.device, dtype=torch.bool)
    min_val = torch.finfo(sim_matrix.dtype).min
    sim_matrix.masked_fill_(mask, min_val)

    # Construct target indices: for each sample, its paired view is the positive.
    target = torch.arange(batch_size, device=embeddings.device)
    target = torch.cat([target + batch_size, target], dim=0)

    # Compute the standard consistency (InfoNCE-style) loss.
    consistency = F.cross_entropy(sim_matrix, target)

    # Compute variance regularization for each set of normalized embeddings.
    var_loss_z1 = variance_regularization(z1_norm)
    var_loss_z2 = variance_regularization(z2_norm)
    var_loss = (var_loss_z1 + var_loss_z2) / 2

    # Combine the consistency loss with the variance regularization.
    return consistency + variance_weight * var_loss


def rsf_loss(pred_risks, times, events, num_trees=100):
    """Random Survival Forest inspired loss function.
    
    Args:
        pred_risks (torch.Tensor): Predicted cumulative hazard [batch_size, num_time_points]
        times (torch.Tensor): Observed times [batch_size]
        events (torch.Tensor): Event indicators [batch_size]
        num_trees (int): Number of bootstrap samples to use
    """
    # Ensure all inputs are on the same device
    device = pred_risks.device
    times = times.to(device)
    events = events.to(device)
    
    batch_size = len(times)
    
    total_loss = 0
    for _ in range(num_trees):
        # Bootstrap sampling
        indices = torch.randint(0, batch_size, (batch_size,), device=device)
        boot_risks = pred_risks[indices]
        boot_times = times[indices]
        boot_events = events[indices]
        
        # Calculate Nelson-Aalen estimator
        sorted_times, sort_idx = torch.sort(boot_times)
        sorted_events = boot_events[sort_idx]
        sorted_risks = boot_risks[sort_idx]
        
        # Compute empirical cumulative hazard
        risk_sets = torch.arange(batch_size, 0, -1, dtype=torch.float32, device=device)
        hazard = sorted_events / risk_sets
        emp_cum_hazard = torch.cumsum(hazard, 0)
        
        # Compare with predicted cumulative hazard
        loss = F.mse_loss(sorted_risks, emp_cum_hazard)
        total_loss += loss
        
    return total_loss / num_trees