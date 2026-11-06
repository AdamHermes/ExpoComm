"""SMAC with observation sight range override. Protocol still needs MAIL audit."""
from smac.env import StarCraft2Env


class LimitedViewSMAC(StarCraft2Env):
    def __init__(self, sight_range=2.0, **kwargs):
        self.observation_sight_range = float(sight_range)
        if self.observation_sight_range <= 0:
            raise ValueError("sight_range must be positive")
        super().__init__(**kwargs)

    def unit_sight_range(self, agent_id):
        return self.observation_sight_range

    def get_agent_active(self):
        return [int(self.get_unit_by_id(i).health > 0) for i in range(self.n_agents)]
