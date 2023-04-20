#####
##### Training hyperparameters
#####

Network = NetLib.ViT

#### Condition: embed_size % num_heads == 0
netparams = NetLib.ViTHP(
  embed_size=96,
  num_heads=8,
  depth=7,
  patch_size=2)

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=5000,
    num_workers=128,
    batch_size=64,
    use_gpu=true,
    reset_every=2,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=600,
    cpuct=2.0,
    prior_temperature=1.0,
    temperature=PLSchedule([0, 20, 30], [1.0, 1.0, 0.3]),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=1.0))

arena = ArenaParams(
  sim=SimParams(
    num_games=128,
    num_workers=128,
    batch_size=128,
    use_gpu=true,
    reset_every=2,
    flip_probability=0.5,
    alternate_colors=true),
  mcts=MctsParams(
    self_play.mcts,
    temperature=ConstSchedule(0.2),
    dirichlet_noise_ϵ=0.05),
  update_threshold=0.05)

learning = LearningParams(
  use_gpu=true,
  use_position_averaging=true,
  samples_weighing_policy=LOG_WEIGHT,
  batch_size=1024,
  loss_computation_batch_size=1024,
  optimiser=Adam(lr=1e-4),
  l2_regularization=1e-4,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=2000,
  num_checkpoints=1)

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=5,
  ternary_rewards=true,
  use_symmetries=true,
  memory_analysis=nothing,
  mem_buffer_size=PLSchedule(
  [      0,        15],
  [400_000, 1_000_000]))

#####
##### Evaluation benchmark
#####

mcts_baseline =
  Benchmark.MctsRollouts(
    MctsParams(
      arena.mcts,
      num_iters_per_turn=1000,
      cpuct=1.))

minmax_baseline = Benchmark.MinMaxTS(
  depth=5,
  τ=0.2,
  amplify_rewards=true)

alphazero_player = Benchmark.Full(arena.mcts)

network_player = Benchmark.NetworkOnly(τ=0.5)

benchmark_sim = SimParams(
  arena.sim;
  num_games=256,
  num_workers=256,
  batch_size=256,
  alternate_colors=false)

benchmark = [
  Benchmark.Duel(alphazero_player, mcts_baseline,   benchmark_sim),
# Benchmark.Duel(alphazero_player, minmax_baseline, benchmark_sim),
  Benchmark.Duel(network_player,   mcts_baseline,   benchmark_sim),
# Benchmark.Duel(network_player,   minmax_baseline, benchmark_sim)
]

#####
##### Wrapping up in an experiment
#####

experiment = Experiment("connect-four",
  GameSpec(), params, Network, netparams, benchmark)