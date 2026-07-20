"""Typed async client for the simulation-engine REST API (port 8003)."""

from pydantic import BaseModel, ConfigDict

from prediction_engine.clients.base import ServiceClient


class SimulationResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    home_win_probability: float
    away_win_probability: float
    draw_probability: float = 0.0
    mean_home_score: float = 0.0
    mean_away_score: float = 0.0
    mean_total: float = 0.0
    mean_margin: float = 0.0
    spread_cover_probabilities: dict[str, float] = {}
    total_over_probabilities: dict[str, float] = {}


class SimulationRun(BaseModel):
    model_config = ConfigDict(extra="ignore")

    simulation_run_id: str
    game_id: str
    status: str = ""
    iterations_completed: int = 0
    converged: bool = False
    result: SimulationResult


class PlayerStatSimulation(BaseModel):
    """One player stat's simulated view: an over grid (count/yardage stats,
    string half-step line keys) or a yes_probability (yes/no stats)."""

    model_config = ConfigDict(extra="ignore")

    distribution: dict[str, float] = {}
    over_probabilities: dict[str, float] = {}
    yes_probability: float | None = None


class PlayerSimulation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    team: str = ""  # "HOME" | "AWAY"
    stats: dict[str, PlayerStatSimulation] = {}


class PlayerDistributions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    simulation_run_id: str
    game_id: str
    players: dict[str, PlayerSimulation] = {}


class SimulationClient(ServiceClient):
    service_name = "simulation-engine"

    async def get_run(self, simulation_run_id: str) -> SimulationRun:
        data = await self.get_data(
            f"/api/v1/sim/simulations/{simulation_run_id}", f"simulation run {simulation_run_id}"
        )
        return SimulationRun.model_validate(data)

    async def get_player_distributions(self, simulation_run_id: str) -> PlayerDistributions:
        """Per-player stat distributions for a run captured with player stats.

        Raises NotFoundError when the run exists but was simulated without
        player capture (the predictor maps that to a 422).
        """
        data = await self.get_data(
            f"/api/v1/sim/simulations/{simulation_run_id}/player-distributions",
            f"player distributions for simulation run {simulation_run_id}",
        )
        return PlayerDistributions.model_validate(data)

    async def health(self) -> bool:
        return await self.is_healthy("/api/v1/sim/health")
