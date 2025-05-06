import operator

'''
Example of using 'between' operation:

shaping_conditions = [
    {'index': 0,
    'operation':'between',
    'threshold': (-0.5, 0.5),
    'reward_modifier': 0.2,
    'description': 'centered on x-axis'},

    {'index': 1,
    'operation':'<',
    'threshold': 0.2,
    'reward_modifier': -0.2,
     'description': 'y position too low'}
]

Example of using RewardShaper class in main script:

from reward_shaping import RewardShaper

# Define shaping conditions for LunarLander
shaping_conditions = [
    {'index': 0, 'operation': '>', 'threshold': 0.5, 'modifier': 0.1, 'description': 'x position > 0.5'},
    {'index': 1, 'operation': '<', 'threshold': 0.2, 'modifier': -0.2, 'description': 'y position < 0.2'},
    {'index': 4, 'operation': '<', 'threshold': -0.2, 'modifier': -0.1, 'description': 'tilted too far left'},
]

reward_shaper = RewardShaper(shaping_conditions)

# Inside your rollout:
obs, info = env.reset()
done = False
episode_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Apply reward shaping
    reward, applied_conditions = reward_shaper.shape(reward, obs)

    # Optional: log condition triggers
    if applied_conditions:
        for cond in applied_conditions:
            print(f"Condition met: {cond['description']} | Value: {cond['value']:.2f} | Modifier: {cond['modifier']:.2f}")

    episode_reward += reward

'''
# Supported comparison operations
OPERATIONS = {
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    'abs>': operator.gt,
    'abs<': operator.lt,
    'abs>=': operator.ge,
    'abs<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne,
    'between': lambda x, bounds: bounds[0] <= x <= bounds[1]
}


def apply_reward_conditions(reward, obs, conditions):
    """
    Applies reward modifications based on provided conditions and logs the effects.

    Args:
        reward (float): Original reward from environment
        obs (np.ndarray or list): Raw observation vector
        conditions (list of dict): Each dict specifies a shaping rule

    Returns:
        tuple: (shaped_reward, list of applied condition logs)
    """
    shaped_reward = reward
    applied_conditions = []

    if isinstance(conditions, dict):
        conditions = [conditions]

    for condition in conditions:
        idx = condition.get('index')
        operation = condition.get('operation')
        threshold = condition.get('threshold')
        reward_modifier = condition.get('reward_modifier')
        description = condition.get('description', '')

        if idx is not None and 0 <= idx < len(obs):
            obs_value = obs[idx]
            op_func = OPERATIONS.get(operation)
            if op_func:
                try:
                    if operation == 'between':
                        if op_func(obs_value, threshold):
                            shaped_reward += reward_modifier
                            applied_conditions.append({
                                'description': description or f'{obs_value} between {threshold}',
                                'obs_value': obs_value,
                                'reward_modifier': reward_modifier
                            })
                    elif operation.startswith('abs'):
                        if op_func(abs(obs_value), threshold):
                            shaped_reward += reward_modifier
                            applied_conditions.append({
                                'description': description or f'abs({obs_value}) {operation} {threshold}',
                                'obs_value': obs_value,
                                'reward_modifier': reward_modifier
                            })
                    else:
                        if op_func(obs_value, threshold):
                            shaped_reward += reward_modifier
                            applied_conditions.append({
                                'description': description or f'{obs_value} {operation} {threshold}',
                                'obs_value': obs_value,
                                'reward_modifier': reward_modifier
                            })
                except Exception as e:
                    print(f"[Warning] Error applying condition {condition}: {e}\n{reward_modifier}")
            else:
                print(f"[Warning] Unsupported operation '{operation}' in condition: {condition}")

    return shaped_reward, applied_conditions


class RewardShaper:
    def __init__(self, conditions=None):
        """
        Initialize with a list of reward shaping conditions.
        Each condition must include 'index', 'operation', 'threshold', 'modifier', and optionally 'description'.
        """
        self.conditions = conditions
    # reset method is a placeholder for any reset logic needed
    # in any environment specific variant of the RewardShaper class

    def reset(self):
        pass

    def shape(self, reward, obs):
        """
        Apply the reward shaping logic to the current reward and observation.
        """
        return apply_reward_conditions(reward, obs, self.conditions)
