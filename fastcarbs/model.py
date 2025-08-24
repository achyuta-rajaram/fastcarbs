from typing import List, Optional

import attr
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from torch import Tensor
from torch.distributions import Normal

# Reuse your exact datatypes/enums so it's a drop-in
from carbs.utils import (
    ObservationInBasic,
    OutstandingSuggestionEstimatorEnum,
    SuggestionInBasic,
    SurrogateModelParams,
)

import fastcarbs_core as core

TorchTensor = torch.Tensor

@attr.s(auto_attribs=True, collect_by_mro=True)
class SurrogateObservationOutputs:
    surrogate_output: Tensor
    surrogate_var: Tensor
    cost_estimate: Tensor
    target_estimate: Tensor
    target_var: Tensor
    success_probability: Tensor
    pareto_surrogate: Optional[TorchTensor] = None
    pareto_estimate: Optional[TorchTensor] = None


class _CppGP:
    def __init__(self, kernel: core.Kernel, noise_var: float = 1e-2, jitter: float = 1e-4):
        self._gp = core.GPRegression(kernel, noise_var, jitter)
        self._fitted = False

    def set_data(self, X: TorchTensor, y: TorchTensor) -> None:
        self._X_cache = X.detach().cpu().double().clone()
        self._y_cache = y.detach().cpu().double().clone()
        self._gp.set_data(self._X_cache.numpy(), self._y_cache.numpy())

    def fit(self, num_steps: int = 1000, lr: float = 1e-4) -> None:
        self._gp.fit(num_steps, lr)
        self._fitted = True

    def predict(self, Xq: TorchTensor, noiseless: bool = True):
        mean_np, var_np = self._gp.predict(Xq.detach().cpu().double().numpy(), noiseless)
        mean = torch.from_numpy(np.asarray(mean_np)).to(torch.float64)
        var = torch.from_numpy(np.asarray(var_np)).to(torch.float64)
        return mean, var

    def sample_marginals(self, Xq: TorchTensor, noiseless: bool = True, seed: int = 0) -> TorchTensor:
        s = self._gp.sample_marginals(Xq.detach().cpu().double().numpy(), noiseless, int(seed))
        return torch.from_numpy(np.asarray(s)).to(torch.float64)

    @property
    def Xy(self):
        return self._X_cache.clone(), self._y_cache.clone()


class SurrogateModel:
    def __init__(self, params: SurrogateModelParams) -> None:
        self.params = params
        self.output_transformer: Optional[QuantileTransformer] = None
        self.cost_transformer: Optional[MinMaxScaler] = None
        self.output_model: Optional[_CppGP] = None
        self.cost_model: Optional[_CppGP] = None
        self.pareto_model: Optional[_CppGP] = None
        self.success_model: Optional[_CppGP] = None
        self.min_logcost: float = float("-inf")
        self.max_logcost: float = float("inf")
        self.min_pareto_logcost: float = float("-inf")
        self.max_pareto_logcost: float = float("inf")
        # caches for refits
        self._X_obs: Optional[TorchTensor] = None
        self._y_obs: Optional[TorchTensor] = None

    def _make_main_kernel(self) -> core.Kernel:
        d = int(self.params.real_dims)
        return core.SumKernel(
            core.Matern32ARD(d, self.params.scale_length, 1.0), core.LinearKernel(d, 1.0)
        )

    def _make_pareto_kernel(self) -> core.Kernel:
        return core.RBF1D(1.0, 1.0)

    def _get_model(self, X: TorchTensor, y: TorchTensor, kernel: Optional[core.Kernel] = None) -> _CppGP:
        if kernel is None:
            kernel = self._make_main_kernel()
        gp = _CppGP(kernel, noise_var=1e-2, jitter=1e-4)
        gp.set_data(X.double(), y.double())
        gp.fit()
        return gp

    def _fit_target_transformers(self, success_observations: List[ObservationInBasic]) -> None:
        raw_outputs = np.array([x.output for x in success_observations], dtype=np.float64)
        n_quantiles = int(np.sqrt(len(success_observations)))
        self.output_transformer = QuantileTransformer(
            output_distribution="normal", n_quantiles=n_quantiles
        )
        self.output_transformer.fit(raw_outputs.reshape(-1, 1))
        log_costs = np.log(np.array([x.cost for x in success_observations], dtype=np.float64))
        self.cost_transformer = MinMaxScaler(feature_range=(-1, 1))
        self.cost_transformer.fit(log_costs.reshape(-1, 1))
        transformed = self.cost_transformer.transform(log_costs.reshape(-1, 1))
        self.min_logcost = float(transformed.min())
        self.max_logcost = float(transformed.max())

    def _target_to_surrogate(self, x: TorchTensor) -> TorchTensor:
        assert self.output_transformer is not None
        x_np = x.cpu().double().view(-1, 1).numpy()
        tx = self.output_transformer.transform(x_np) * self.params.better_direction_sign
        return torch.from_numpy(tx).to(torch.float64).view(*x.shape)

    def _surrogate_to_target(self, x: TorchTensor) -> TorchTensor:
        assert self.output_transformer is not None
        x_np = x.detach().cpu().double().view(-1, 1).numpy()
        inv = self.output_transformer.inverse_transform(
            x_np * self.params.better_direction_sign
        )
        return torch.from_numpy(inv).to(torch.float64).view(*x.shape)

    def _cost_to_logcost(self, x: TorchTensor) -> TorchTensor:
        assert self.cost_transformer is not None
        x_np = torch.log(x.detach().cpu().double().view(-1, 1)).numpy()
        tx = self.cost_transformer.transform(x_np)
        return torch.from_numpy(tx).to(torch.float64).view(*x.shape)

    def _logcost_to_cost(self, x: TorchTensor) -> TorchTensor:
        assert self.cost_transformer is not None
        inv = self.cost_transformer.inverse_transform(
            x.detach().cpu().double().view(-1, 1).numpy()
        )
        return torch.from_numpy(np.exp(inv)).to(torch.float64).view(*x.shape)

    # ---- public API used by CARBS ----
    def fit_observations(self, success_observations: List[ObservationInBasic]) -> None:
        self._fit_target_transformers(success_observations)
        X = (
            torch.stack([x.real_number_input for x in success_observations])
            .detach()
            .to(torch.float64)
        )
        y = self._target_to_surrogate(
            torch.tensor([x.output for x in success_observations], dtype=torch.float64)
        )
        self._X_obs, self._y_obs = X.clone(), y.clone()
        self.output_model = self._get_model(X, y)
        logc = self._cost_to_logcost(
            torch.tensor([x.cost for x in success_observations], dtype=torch.float64)
        )
        self.cost_model = self._get_model(X, logc)

    def fit_suggestions(self, outstanding_suggestions: List[SuggestionInBasic]) -> None:
        if len(outstanding_suggestions) == 0:
            return
        assert self.output_model is not None and self._X_obs is not None and self._y_obs is not None
        X_new = (
            torch.stack([x.real_number_input for x in outstanding_suggestions])
            .detach()
            .to(torch.float64)
        )
        if (
            self.params.outstanding_suggestion_estimator
            == OutstandingSuggestionEstimatorEnum.MEAN
        ):
            mu, _ = self.output_model.predict(X_new, noiseless=True)
            y_new = mu
        elif (
            self.params.outstanding_suggestion_estimator
            == OutstandingSuggestionEstimatorEnum.THOMPSON
        ):
            y_new = self.output_model.sample_marginals(X_new, noiseless=True)
        else:
            raise RuntimeError("Unsupported outstanding suggestion estimator")
        X_aug = torch.cat([self._X_obs, X_new], dim=0)
        y_aug = torch.cat([self._y_obs, y_new], dim=0)
        # Refit a fresh model to augmented data
        self.output_model = self._get_model(X_aug, y_aug)

    def fit_pareto_set(self, pareto_observations: List[ObservationInBasic]) -> None:
        if len(pareto_observations) == 0:
            self.pareto_model = None
            return
        costs = torch.tensor([x.cost for x in pareto_observations], dtype=torch.float64)
        logc = self._cost_to_logcost(costs).view(-1, 1)  # 1D input
        outs = self._target_to_surrogate(
            torch.tensor([x.output for x in pareto_observations], dtype=torch.float64)
        )
        gp = _CppGP(self._make_pareto_kernel(), noise_var=1e-2, jitter=1e-4)
        gp.set_data(logc, outs)
        gp.fit(num_steps=1000, lr=1e-2)  # 1D => fewer steps suffice
        self.pareto_model = gp
        self.min_pareto_logcost = float(logc.min().item())
        self.max_pareto_logcost = float(logc.max().item())

    def get_pareto_surrogate_for_cost(self, cost: float) -> float:
        assert self.pareto_model is not None
        c = torch.tensor([cost], dtype=torch.float64)
        lc = self._cost_to_logcost(c)
        lc = torch.clamp(lc, min=self.min_pareto_logcost, max=self.max_pareto_logcost)
        mu, _ = self.pareto_model.predict(lc.view(-1, 1), noiseless=True)
        return float(mu.item())

    def fit_failures(
        self,
        success_observations: List[ObservationInBasic],
        failure_observations: List[ObservationInBasic],
    ) -> None:
        n_s, n_f = len(success_observations), len(failure_observations)
        if n_s == 0 or n_f == 0:
            self.success_model = None
            return
        all_obs = success_observations + failure_observations
        X = (
            torch.stack([x.real_number_input for x in all_obs]).detach().to(torch.float64)
        )
        labels = torch.tensor([-1] * n_s + [1] * n_f, dtype=torch.float64)
        gp = _CppGP(self._make_main_kernel(), noise_var=1e-2, jitter=1e-4)
        gp.set_data(X, labels)
        gp.fit(num_steps=1000, lr=1e-2)
        self.success_model = gp

    def _get_success_prob(self, X: TorchTensor) -> TorchTensor:
        if self.success_model is None:
            return torch.ones((X.shape[0],), dtype=torch.float64)
        mu, var = self.success_model.predict(X.to(torch.float64), noiseless=True)
        prior = Normal(mu, var)
        return prior.cdf(torch.zeros_like(mu, dtype=torch.float64))

    @torch.no_grad()
    def observe_surrogate(self, samples_in_basic: TorchTensor) -> SurrogateObservationOutputs:
        assert self.output_model is not None and self.cost_model is not None
        Xq = samples_in_basic.to(torch.float64)
        mu_surr, var_surr = self.output_model.predict(Xq, noiseless=True)
        mu_logc, var_logc = self.cost_model.predict(Xq, noiseless=True)
        if self.pareto_model is not None:
            lc = torch.clamp(
                mu_logc, min=self.min_pareto_logcost, max=self.max_pareto_logcost
            )
            pareto_mu, pareto_var = self.pareto_model.predict(lc.view(-1, 1), noiseless=True)
            pareto_est = self._surrogate_to_target(pareto_mu)
        else:
            pareto_mu = None
            pareto_est = None

        success_p = self._get_success_prob(Xq)
        cost_est = self._logcost_to_cost(mu_logc)
        target_est = self._surrogate_to_target(mu_surr)
        # target variance (for logging): approximate via +/- 1 std mapping through inverse
        std_s = torch.sqrt(torch.clamp(var_surr, min=1e-12))
        tgt_hi = self._surrogate_to_target(mu_surr + std_s)
        tgt_lo = self._surrogate_to_target(mu_surr - std_s)
        tgt_var = torch.square((tgt_hi - tgt_lo) / 2.0)

        return SurrogateObservationOutputs(
            surrogate_output=mu_surr,
            surrogate_var=var_surr,
            cost_estimate=cost_est,
            pareto_surrogate=pareto_mu,
            pareto_estimate=pareto_est,
            target_estimate=target_est,
            target_var=tgt_var,
            success_probability=success_p,
        )


