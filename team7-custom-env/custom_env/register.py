from gymnasium.envs.registration import register

register(
         id = 'Team7-v0',
         entry_point = 'custom_env.envs:MyEnv',
)