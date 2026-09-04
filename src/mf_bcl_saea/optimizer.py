"""MF-BCL-SAEA reconstruction loop with a high-fidelity-only archive."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .controller import BayesianOperatorController, OperatorAction
from .features import normalized_design, phase_sensitivity, phase_transition_descriptors
from .lce import LCEFidelityEvaluator
from .metrics import hypervolume_2d
from .nsga2 import rank_and_crowding, sbx_pair, polynomial_mutation, tournament
from .schema import AblationFlags, Evaluation, Fidelity, PaperProtocol
from .surrogate import ResidualMultiFidelityEnsemble


@dataclass
class GenerationLog:
    generation: int
    high_fidelity_evaluations: int
    archive_size: int
    selected_for_upgrade: int
    controller_action: OperatorAction
    surrogate_source: str
    reward: float


@dataclass
class OptimizationResult:
    archive: list[Evaluation]
    all_evaluations: list[Evaluation]
    logs: list[GenerationLog]
    paper_equivalent: bool
    limitations: list[str] = field(default_factory=list)

    @property
    def objective_front(self) -> np.ndarray:
        return np.asarray([item.objectives for item in self.archive]) if self.archive else np.empty((0, 2))

    @property
    def high_fidelity_evaluations(self) -> int:
        return sum(item.fidelity is Fidelity.HIGH for item in self.all_evaluations)


class MFBCLSAEA:
    """Operational implementation of the algorithmic loop in the 1.0 manuscript.

    The published text does not specify the initial DOE, L/M scheduling, surrogate
    retraining schedule, score normalization or stopping condition behind 136 HF
    calls. The transparent defaults below are therefore reconstruction choices,
    not recovered paper settings. They can be overridden after source recovery.
    """

    def __init__(
        self,
        evaluator: LCEFidelityEvaluator,
        protocol: PaperProtocol = PaperProtocol(),
        flags: AblationFlags = AblationFlags(),
        seed: int = 0,
        hypervolume_reference: np.ndarray | None = None,
        surrogate_refit_interval: int = 5,
    ) -> None:
        self.evaluator = evaluator
        self.protocol = protocol
        self.flags = flags
        self.rng = np.random.default_rng(seed)
        self.surrogate = ResidualMultiFidelityEnsemble(evaluator.bounds, protocol, seed)
        self.controller = BayesianOperatorController(protocol, flags, seed)
        self.hypervolume_reference = None if hypervolume_reference is None else np.asarray(hypervolume_reference, dtype=float)
        self.surrogate_refit_interval = surrogate_refit_interval

    def _evaluate(self, x: np.ndarray, fidelity: Fidelity, evaluations: list[Evaluation]) -> Evaluation:
        result = self.evaluator.evaluate(x, fidelity)
        evaluations.append(result)
        self.surrogate.observe(result)
        return result

    def _state(self, x: np.ndarray, rank: float, crowding: float, generation: int, hv_progress: float, hf_used: int) -> np.ndarray:
        design = normalized_design(x, self.evaluator.bounds.lower, self.evaluator.bounds.upper)
        descriptors = phase_transition_descriptors(x)[0] if self.flags.phase_conditioned_state else np.zeros(6)
        crowd_value = 1.0 if np.isinf(crowding) else float(np.clip(crowding, 0.0, 1.0))
        return np.hstack((design, descriptors, [rank, crowd_value, generation / self.protocol.maximum_generations, hv_progress, hf_used / self.protocol.high_fidelity_budget])).astype(np.float32)

    def _score_upgrades(self, x: np.ndarray, rank: np.ndarray, crowding: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
        pareto_potential = 1.0 - rank / max(np.max(rank), 1)
        finite_crowding = np.where(np.isfinite(crowding), crowding, np.nanmax(np.where(np.isfinite(crowding), crowding, 0.0)) + 1.0)
        pareto_potential += self._unit_interval(finite_crowding)
        uncertainty_score = self._unit_interval(uncertainty)
        phase_score = self._unit_interval(phase_sensitivity(phase_transition_descriptors(x))) if self.flags.phase_weighted_upgrade else np.zeros(len(x))
        w1, w2, w3 = self.protocol.update_score_weights
        return w1 * self._unit_interval(pareto_potential) + w2 * uncertainty_score + w3 * phase_score

    @staticmethod
    def _unit_interval(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        span = values.max() - values.min()
        return np.zeros_like(values) if span < 1e-12 else (values - values.min()) / span

    def _archive(self, evaluations: list[Evaluation]) -> list[Evaluation]:
        candidates = [item for item in evaluations if item.fidelity is Fidelity.HIGH and item.feasible]
        if not candidates:
            return []
        objectives = np.asarray([item.objectives for item in candidates])
        rank, _ = rank_and_crowding(objectives, np.zeros((len(candidates), 1)))
        return [item for item, level in zip(candidates, rank, strict=True) if level == 0]

    def _reward(self, archive: list[Evaluation], previous_hv: float) -> tuple[float, float]:
        if not archive:
            return 0.0, previous_hv
        objectives = np.asarray([item.objectives for item in archive])
        if self.hypervolume_reference is not None:
            current_hv = hypervolume_2d(objectives, self.hypervolume_reference)
            return current_hv - previous_hv, current_hv
        # Without the unpublished reference point, a normalized-HF-improvement
        # proxy is used only to drive the controller, never reported as HV.
        proxy = -float(np.mean(objectives))
        return proxy - previous_hv, proxy

    def _offspring(self, population: np.ndarray, objectives: np.ndarray, constraints: np.ndarray, action: OperatorAction) -> np.ndarray:
        rank, crowding = rank_and_crowding(objectives, constraints)
        children: list[np.ndarray] = []
        indices = np.arange(len(population))
        while len(children) < self.protocol.population_size:
            a = population[tournament(indices, rank, crowding, self.rng)]
            b = population[tournament(indices, rank, crowding, self.rng)]
            child_a, child_b = sbx_pair(a, b, self.evaluator.bounds.lower, self.evaluator.bounds.upper, action.crossover_probability, action.crossover_eta, self.rng)
            children.append(polynomial_mutation(child_a, self.evaluator.bounds.lower, self.evaluator.bounds.upper, action.mutation_probability, action.mutation_eta, self.rng))
            if len(children) < self.protocol.population_size:
                children.append(polynomial_mutation(child_b, self.evaluator.bounds.lower, self.evaluator.bounds.upper, action.mutation_probability, action.mutation_eta, self.rng))
        return np.asarray(children)

    def run(self) -> OptimizationResult:
        population = self.rng.uniform(self.evaluator.bounds.lower, self.evaluator.bounds.upper, size=(self.protocol.population_size, 14))
        all_evaluations: list[Evaluation] = []
        logs: list[GenerationLog] = []
        previous_hv = 0.0
        last_task = -1
        for generation in range(self.protocol.maximum_generations):
            low = [self._evaluate(x, Fidelity.LOW, all_evaluations) for x in population]
            low_objectives = np.asarray([item.objectives for item in low])
            low_constraints = np.asarray([item.constraints for item in low])
            if generation % self.surrogate_refit_interval == 0:
                self.surrogate.fit()
            prediction = self.surrogate.predict(population, low_objectives)
            predicted_constraints = low_constraints
            rank, crowding = rank_and_crowding(prediction.mean, predicted_constraints)
            uncertainty = np.linalg.norm(prediction.std, axis=1) if self.flags.uncertainty else np.zeros(len(population))
            scores = self._score_upgrades(population, rank, crowding, uncertainty)
            remaining = self.protocol.high_fidelity_budget - sum(item.fidelity is Fidelity.HIGH for item in all_evaluations)
            upgrade_indices = np.argsort(-scores)[: min(self.protocol.high_fidelity_upgrades_per_generation, remaining)]
            for index in upgrade_indices:
                self._evaluate(population[index], Fidelity.MEDIUM, all_evaluations)
                self._evaluate(population[index], Fidelity.HIGH, all_evaluations)
            archive = self._archive(all_evaluations)
            reward, current_hv = self._reward(archive, previous_hv)
            elite = int(np.argmin(rank))
            state = self._state(population[elite], float(rank[elite]), float(crowding[elite]), generation, current_hv - previous_hv, self._hf_count(all_evaluations))
            self.controller.observe_reward(state, reward, float(phase_sensitivity(phase_transition_descriptors(population[elite]))[0]), max(reward, 0.0), float(uncertainty[elite]))
            action = self.controller.sample(state) if self.flags.continual_controller else self.controller.default_action()
            if (generation + 1) % self.protocol.controller_update_interval_generations == 0:
                self.controller.update()
            task = min(self.protocol.pseudo_task_count - 1, generation * self.protocol.pseudo_task_count // self.protocol.maximum_generations)
            if task != last_task and last_task >= 0:
                self.controller.snapshot_fisher()
            last_task = task
            logs.append(GenerationLog(generation, self._hf_count(all_evaluations), len(archive), len(upgrade_indices), action, prediction.source, reward))
            previous_hv = current_hv
            population = self._offspring(population, prediction.mean, predicted_constraints, action)
            if remaining <= 0:
                # Continue evolution only when an explicitly different stopping
                # rule is supplied; paper's 136-HF trigger is not disclosed.
                break
        paper_equivalent = all(item.metadata.get("evidence") == "registered_external_abaqus" for item in all_evaluations)
        limitations = []
        if not paper_equivalent:
            limitations.append("Uses a non-evidentiary fixture; no published numerical claim is supported.")
        limitations.append("Exact source-only DOE, constraints, reference front and early-stopping rule remain unregistered.")
        return OptimizationResult(self._archive(all_evaluations), all_evaluations, logs, paper_equivalent, limitations)

    @staticmethod
    def _hf_count(evaluations: list[Evaluation]) -> int:
        return sum(item.fidelity is Fidelity.HIGH for item in evaluations)
