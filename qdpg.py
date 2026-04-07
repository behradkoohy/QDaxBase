import os

from IPython.display import clear_output
import functools
import time

import jax
import jax.numpy as jnp

from qdax.core.containers.archive import score_euclidean_novelty
from qdax.core.emitters.dpg_emitter import DiversityPGConfig
from qdax.core.emitters.qdpg_emitter import QDPGEmitter, QDPGEmitterConfig
from qdax.core.emitters.qpg_emitter import QualityPGConfig
from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
import qdax.tasks.brax as environments
from qdax.tasks.brax.env_creators import scoring_function_brax_envs as scoring_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.utils.plotting import plot_map_elites_results

from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.utils.metrics import CSVLogger, default_qd_metrics


from tqdm import trange
#@title QD Training Definitions Fields
#@markdown ---
env_name = 'humanoid_uni'#@param['ant_uni', 'hopper_uni', 'walker_uni', 'halfcheetah_uni', 'humanoid_uni', 'ant_omni', 'humanoid_omni']
episode_length = 1000 #@param {type:"integer"}
num_iterations = 250 #@param {type:"integer"}
seed = 5 #@param {type:"integer"}
policy_hidden_layer_sizes = (256, 256) #@param {type:"raw"}
iso_sigma = 0.005 #@param {type:"number"}
line_sigma = 0.05 #@param {type:"number"}
num_init_cvt_samples = 50000 #@param {type:"integer"}
num_centroids = 1024 #@param {type:"integer"}
min_descriptor = 0. #@param {type:"number"}
max_descriptor = 1.0 #@param {type:"number"}

# mutations size
quality_pg_batch_size = 4 #@param {type:"integer"}
diversity_pg_batch_size = 3 #@param {type:"integer"}
ga_batch_size = 3 #@param {type:"integer"}

env_batch_size = quality_pg_batch_size + diversity_pg_batch_size + ga_batch_size

# TD3 params
replay_buffer_size = 1000000 #@param {type:"number"}
critic_hidden_layer_size = (256, 256) #@param {type:"raw"}
critic_learning_rate = 3e-4 #@param {type:"number"}
greedy_learning_rate = 3e-4 #@param {type:"number"}
policy_learning_rate = 1e-3 #@param {type:"number"}
noise_clip = 0.5 #@param {type:"number"}
policy_noise = 0.2 #@param {type:"number"}
discount = 0.99 #@param {type:"number"}
reward_scaling = 1.0 #@param {type:"number"}
transitions_batch_size = 256 #@param {type:"number"}
soft_tau_update = 0.005 #@param {type:"number"}
num_critic_training_steps = 300 #@param {type:"number"}
num_pg_training_steps = 100 #@param {type:"number"}
policy_delay = 2 #@param {type:"number"}

archive_acceptance_threshold = 0.1 #@param {type:"number"}
archive_max_size = 10000 #@param {type:"integer"}

num_nearest_neighb = 5 #@param {type:"integer"}
novelty_scaling_ratio = 1.0 #@param {type:"number"}
#@markdown ---

# Init environment
env = environments.create(env_name, episode_length=episode_length, backend="mjx")
reset_fn = jax.jit(env.reset)

# Init a random key
key = jax.random.key(seed)

# Init policy network
policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
policy_network = MLP(
    layer_sizes=policy_layer_sizes,
    kernel_init=jax.nn.initializers.lecun_uniform(),
    final_activation=jnp.tanh,
)

# Init population of controllers
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=env_batch_size)
fake_batch = jnp.zeros(shape=(env_batch_size, env.observation_size))
init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

# Define the function to play a step with the policy in the environment
def play_step_fn(
    env_state,
    policy_params,
    key,
):
    """
    Play an environment step and return the updated state and the transition.
    """

    actions = policy_network.apply(policy_params, env_state.obs)

    state_desc = env_state.info["state_descriptor"]
    next_state = env.step(env_state, actions)

    transition = QDTransition(
        obs=env_state.obs,
        next_obs=next_state.obs,
        rewards=next_state.reward,
        dones=next_state.done,
        actions=actions,
        truncations=next_state.info["truncation"],
        state_desc=state_desc,
        next_state_desc=next_state.info["state_descriptor"],
    )

    return next_state, policy_params, key, transition

# Prepare the scoring function
descriptor_extraction_fn = environments.descriptor_extractor[env_name]
scoring_fn = functools.partial(
    scoring_function,
    episode_length=episode_length,
    play_reset_fn=reset_fn,
    play_step_fn=play_step_fn,
    descriptor_extractor=descriptor_extraction_fn,
)

# Get minimum reward value to make sure qd_score are positive
reward_offset = environments.reward_offset[env_name]

# Define a metrics function
metrics_function = functools.partial(
    default_qd_metrics,
    qd_offset=reward_offset * episode_length,
)

# Define the Quality PG emitter config
qpg_emitter_config = QualityPGConfig(
    env_batch_size=quality_pg_batch_size,
    batch_size=transitions_batch_size,
    critic_hidden_layer_size=critic_hidden_layer_size,
    critic_learning_rate=critic_learning_rate,
    actor_learning_rate=greedy_learning_rate,
    policy_learning_rate=policy_learning_rate,
    noise_clip=noise_clip,
    policy_noise=policy_noise,
    discount=discount,
    reward_scaling=reward_scaling,
    replay_buffer_size=replay_buffer_size,
    soft_tau_update=soft_tau_update,
    num_critic_training_steps=num_critic_training_steps,
    num_pg_training_steps=num_pg_training_steps,
    policy_delay=policy_delay,
)

# Define the Diversity PG emitter config
dpg_emitter_config = DiversityPGConfig(
    env_batch_size=diversity_pg_batch_size,
    batch_size=transitions_batch_size,
    critic_hidden_layer_size=critic_hidden_layer_size,
    critic_learning_rate=critic_learning_rate,
    actor_learning_rate=greedy_learning_rate,
    policy_learning_rate=policy_learning_rate,
    noise_clip=noise_clip,
    policy_noise=policy_noise,
    discount=discount,
    reward_scaling=reward_scaling,
    replay_buffer_size=replay_buffer_size,
    soft_tau_update=soft_tau_update,
    num_critic_training_steps=num_critic_training_steps,
    num_pg_training_steps=num_pg_training_steps,
    policy_delay=policy_delay,
    archive_acceptance_threshold=archive_acceptance_threshold,
    archive_max_size=archive_max_size,
)

# Define the QDPG Emitter config
qdpg_emitter_config = QDPGEmitterConfig(
    qpg_config=qpg_emitter_config,
    dpg_config=dpg_emitter_config,
    iso_sigma=iso_sigma,
    line_sigma=line_sigma,
    ga_batch_size=ga_batch_size,
)

# Get the emitter
score_novelty = jax.jit(
    functools.partial(
        score_euclidean_novelty,
        num_nearest_neighb=num_nearest_neighb,
        scaling_ratio=novelty_scaling_ratio,
    )
)

# define the QDPG emitter
qdpg_emitter = QDPGEmitter(
    config=qdpg_emitter_config,
    policy_network=policy_network,
    env=env,
    score_novelty=score_novelty,
)

# Instantiate MAP Elites
map_elites = MAPElites(
    scoring_function=scoring_fn,
    emitter=qdpg_emitter,
    metrics_function=metrics_function,
)

# Compute the centroids
key, subkey = jax.random.split(key)
centroids = compute_cvt_centroids(
    num_descriptors=env.descriptor_length,
    num_init_cvt_samples=num_init_cvt_samples,
    num_centroids=num_centroids,
    minval=min_descriptor,
    maxval=max_descriptor,
    key=subkey,
)

# compute initial repertoire
key, subkey = jax.random.split(key)
repertoire, emitter_state, init_metrics = map_elites.init(
    init_variables, centroids, subkey
)

log_period = 1
num_loops = num_iterations // log_period

# Initialize metrics
metrics = {key: jnp.array([]) for key in ["iteration", "qd_score", "coverage", "max_fitness", "time"]}

# Set up init metrics
init_metrics = jax.tree.map(lambda x: jnp.array([x]) if x.shape == () else x, init_metrics)
init_metrics["iteration"] = jnp.array([0], dtype=jnp.int32)
init_metrics["time"] = jnp.array([0.0])  # No time recorded for initialization

# Convert init_metrics to match the metrics dictionary structure
metrics = jax.tree.map(lambda metric, init_metric: jnp.concatenate([metric, init_metric], axis=0), metrics, init_metrics)

# Initialize CSV logger
csv_logger = CSVLogger(
    f"qdpg-logs-{seed}.csv",
    header=list(metrics.keys())
)

# Main loop
map_elites_scan_update = map_elites.scan_update
for i in trange(num_loops):
    start_time = time.time()
    (
        repertoire,
        emitter_state,
        key,
    ), current_metrics = jax.lax.scan(
        map_elites_scan_update,
        (repertoire, emitter_state, key),
        (),
        length=log_period,
    )
    timelapse = time.time() - start_time

    # Metrics
    current_metrics["iteration"] = jnp.arange(1+log_period*i, 1+log_period*(i+1), dtype=jnp.int32)
    current_metrics["time"] = jnp.repeat(timelapse, log_period)
    metrics = jax.tree.map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

    # Log
    csv_logger.log(jax.tree.map(lambda x: x[-1], metrics))