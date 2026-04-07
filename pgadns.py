import os

from IPython.display import clear_output
import functools
import time

import jax
import jax.numpy as jnp

# from qdax.core.map_elites import MAPElites
# from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.populations.DominatedNS import AdaptivePopulation

import qdax.tasks.brax as environments
# from qdax.tasks.brax.env_creators import scoring_function_brax_envs as scoring_function
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs as scoring_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.utils.plotting import plot_map_elites_results

from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.utils.metrics import CSVLogger, default_qd_metrics

from tqdm import trange

#@title QD Training Definitions Fields
#@markdown ---
env_name = 'ant_uni'#@param['ant_uni', 'hopper_uni', 'walker_uni', 'halfcheetah_uni', 'humanoid_uni', 'ant_omni', 'humanoid_omni']
episode_length = 1000 #@param {type:"integer"}
num_iterations = 250 #@param {type:"integer"}
seed = 1 #@param {type:"integer"}
policy_hidden_layer_sizes = (256, 256) #@param {type:"raw"}
iso_sigma = 0.005 #@param {type:"number"}
line_sigma = 0.05 #@param {type:"number"}
# num_init_cvt_samples = 50000 #@param {type:"integer"}
# num_centroids = 1024 #@param {type:"integer"}
# DNS Parameters
max_size = 500 #@param {type:"integer"} # Size of the unstructured archive
knn = 10 #@param {type:"integer"} # K-Nearest Neighbors for domination check

min_descriptor = 0. #@param {type:"number"}
max_descriptor = 1.0 #@param {type:"number"}

proportion_mutation_ga = 0.5 #@param {type:"number"}

# TD3 params
env_batch_size = 10 #@param {type:"number"}
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
# soft_tau_update = 0.005 #@param {type:"number"}
soft_tau_update = 0.005 #@param {type:"number"}
num_critic_training_steps = 300 #@param {type:"number"}
num_pg_training_steps = 100 #@param {type:"number"}
policy_delay = 2 #@param {type:"number"}
#@markdown ---


# --- PASTE THIS CLASS INTO pgadns.py (Overrides the previous MAPElites definition) ---
from typing import Any, Callable, Tuple
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.populations.Population import Population
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, Metrics, RNGKey

class MAPElites:
    """Core elements of the MAP-Elites algorithm (DNS Version - Fixed for PGAME)."""

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[Population], Metrics],
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function

    @functools.partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        population: Population,
        genotypes: Genotype,
        random_key: RNGKey,
    ) -> Tuple[Population, EmitterState | None, RNGKey]:
        """
        Initialize a Map-Elites population with an initial population of genotypes.
        """
        # score initial genotypes
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            key=random_key, # <--- FIXED: Changed from 'random_key' to 'key'
            repertoire=population,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )
        print(f'[DEBUG] init descriptors shape: {descriptors.shape} with content: \n {descriptors}')
        return population, emitter_state, random_key

    @functools.partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        population: Population,
        emitter_state: EmitterState | None,
        random_key: RNGKey,
    ) -> Tuple[Population, EmitterState | None, Metrics, RNGKey]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        """
        # generate offsprings with the emitter
        genotypes, extra_info, random_key = self._emitter.emit(
            population, emitter_state, random_key
        )

        # scores the offsprings
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        # add genotypes in the population
        population = population.add(genotypes, descriptors, fitnesses, extra_scores)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=population,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores={**extra_scores, **extra_info},
        )
        # update the metrics
        metrics = self._metrics_function(population)

        return population, emitter_state, metrics, random_key

    @functools.partial(jax.jit, static_argnames=("self",))
    def scan_update(
        self,
        carry: Tuple[Population, EmitterState | None, RNGKey],
        unused: Any,
    ) -> Tuple[Tuple[Population, EmitterState | None, RNGKey], Metrics]:
        """Rewrites the update function in a way that makes it compatible with jax.lax.scan."""
        population, emitter_state, random_key = carry
        (
            population,
            emitter_state,
            metrics,
            random_key,
        ) = self.update(
            population,
            emitter_state,
            random_key,
        )

        return (population, emitter_state, random_key), metrics


env = environments.create(env_name, episode_length=episode_length, backend='mjx')
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
    behavior_descriptor_extractor=descriptor_extraction_fn,
)

# Get minimum reward value to make sure qd_score are positive
reward_offset = environments.reward_offset[env_name]

# Define a metrics function
metrics_function = functools.partial(
    default_qd_metrics,
    qd_offset=reward_offset * episode_length,
)

# Define the PG-emitter config
pga_emitter_config = PGAMEConfig(
    env_batch_size=env_batch_size,
    batch_size=transitions_batch_size,
    proportion_mutation_ga=proportion_mutation_ga,
    critic_hidden_layer_size=critic_hidden_layer_size,
    critic_learning_rate=critic_learning_rate,
    greedy_learning_rate=greedy_learning_rate,
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

# Get the emitter
variation_fn = functools.partial(
    isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
)

pg_emitter = PGAMEEmitter(
    config=pga_emitter_config,
    policy_network=policy_network,
    env=env,
    variation_fn=variation_fn,
)

# Instantiate MAP Elites
map_elites = MAPElites(
    scoring_function=scoring_fn,
    emitter=pg_emitter,
    metrics_function=metrics_function,
)

# # Compute the centroids
# key, subkey = jax.random.split(key)
# centroids = compute_cvt_centroids(
#     num_descriptors=env.descriptor_length,
#     num_init_cvt_samples=num_init_cvt_samples,
#     num_centroids=num_centroids,
#     minval=min_descriptor,
#     maxval=max_descriptor,
#     key=subkey,
# )
#
# # compute initial repertoire
# key, subkey = jax.random.split(key)
# repertoire, emitter_state, init_metrics = map_elites.init(
#     init_variables, centroids, subkey
# )
# --- DNS INITIALIZATION ---

# 1. Score the initial genotypes manually to get data for the population init
key, subkey = jax.random.split(key)
fitnesses, descriptors, extra_scores, subkey = scoring_fn(init_variables, subkey)

# 2. Initialize the Unstructured Adaptive Population (DNS Archive)
# Note: We pass 'extra_scores' as 'observations' because AdaptivePopulation
# expects a structure to store auxiliary data (like transitions for the replay buffer).
repertoire = AdaptivePopulation.init(
    genotypes=init_variables,
    fitnesses=fitnesses,
    descriptors=descriptors,
    observations=extra_scores,
    max_size=max_size,
    k=knn,
)

# 3. Initialize MAP-Elites with the prepared population
# The init method signature changes here to accept the population object
repertoire, emitter_state, init_metrics = map_elites.init(
    repertoire, init_variables, subkey
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
    f"pgame-logs-{seed}.csv",
    header=list(metrics.keys())
)

print('Starting Main Loop')
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