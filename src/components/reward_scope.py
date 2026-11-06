"""Reward layout shared by replay, runner and learners."""


def is_global_reward(args):
    scope = getattr(args, "reward_scope", "legacy")
    if scope == "legacy":
        return args.env == "struct_marl"
    if scope not in ("team", "individual"):
        raise ValueError("reward_scope must be legacy, team or individual")
    return scope == "team"
