---
title: Simulating 1,000,000 Drops of Rain or GPU Terrain Erosion in Rust
---

So I've wanted to write this project up for a while, but clearly I've
procrastinated it for the last couples of years... Anyhow. Picture
this, the year is 2023. I had just graduated and while looking for
jobs, I decided to make a game. At this point Cities Skylines 2 had
just come out with quite a disastrous launch, and I wanted to make a
city builder that was inspired by the traffic system of Cities
Skylines 1, and the grid based simplicity of the childhood classic
Transport Tycoon (well, more like OpenTTD in my case. I'm not quite
old enough for the original).

Now clearly based on the article, I did not get very far on my city
builder. Instead what I got fascinated by was, terrain generation. More
specifically, I wanted to generate realistic looking terrain, that
actually looks like something from Google Earth. Ultimately, I fell
into the rabbit hole of hydraulic erosion, and tried to make droplet
erosion run in real-(ish) time.

<!-- In this post, I want to explain how I simulate hydraulic erosion for terrain generation in my city builder, and why I chose to run the whole process on the GPU. The goal of this article is not just to show some shader code, but to build up the theory behind erosion first, then connect that theory to the concrete Rust and WebGPU implementation. I also want to show how this fits into the larger world generation pipeline, from procedural noise all the way to a mesh you can actually render. -->

<!-- Raw procedural noise is good at producing interesting shapes, but it usually lacks the sense that the terrain has actually been shaped by a physical process. You get mountains, hills, and valleys, but not the kind of drainage patterns, channels, sediment buildup, and weathered landforms that make terrain feel believable. Hydraulic erosion fixes that by treating terrain not just as a sampled function, but as something that changes over time in response to moving water. -->

<!-- This article is going to be fairly technical. I want to start from the physical intuition, reduce that into a discrete simulation over a heightmap, then explain why that kind of workload maps well to compute shaders. After that, I will go through the actual implementation in Rust and WGSL. -->

## Outline

## Why noise alone is not enough

- explain the strengths and weaknesses of procedural noise for terrain generation
- show why raw noise tends to produce terrain that looks synthetic even when it looks varied
- introduce the idea that believable terrain needs transport processes, not just height variation
- motivate hydraulic erosion as a way to create river basins, gullies, sediment fans, and coherent downhill structure
- frame the main idea of the article

## What hydraulic erosion is actually modelling

- define the terrain as a scalar field `h(x, y)`
- explain that water moves downhill according to local slope
- describe sediment as material carried by moving water
- explain that droplets have a carrying capacity based on slope, speed, and water content
- introduce the core rules

1. water moving downhill can erode the ground
2. water can only carry a limited amount of sediment
3. if it carries too much, it deposits material
4. if it evaporates or slows down, it tends to deposit

- discuss why these simple rules are enough to produce realistic looking terrain features

## From physical process to discrete simulation

- explain that the implementation is not a full fluid simulation
- justify the droplet based approximation as a practical compromise
- compare droplet erosion to more expensive PDE based or shallow water methods
- explain the benefits of droplets

1. easy to reason about
2. easy to parallelize
3. good enough for terrain synthesis
4. naturally maps to a heightmap representation

- mention the downsides too

1. not fully physically accurate
2. can introduce noise and artifacts
3. can be nondeterministic when parallelized aggressively

## The state of a single droplet

- introduce the droplet state used in the simulation
- explain each field and why it matters

1. position
2. direction
3. speed
4. water amount
5. sediment amount
6. radius

- discuss radius as a way to spread erosion and deposition over an area instead of a single point
- connect this to the implementation struct in `src/world_gen/erosion.rs`

## How one erosion step works

- present the droplet update loop at a high level before showing any code
- walk through the steps in order

1. sample the local neighborhood
2. estimate downhill direction from height differences
3. normalize and update the droplet direction
4. move the droplet
5. update water and speed
6. compute sediment carrying capacity
7. choose whether to erode or deposit
8. stop if the droplet leaves the map, dries out, or loses momentum

- include a small pseudocode block for the erosion step
- explain that the whole simulation is just this loop applied to many droplets

## Neighbourhood sampling and slope estimation

- explain why the shader does not just look at one downhill neighbor
- describe the circular neighborhood around the droplet radius
- explain how local height differences contribute to a direction vector
- discuss the role of gravity and directional inertia in the weighted direction update
- explain why this produces smoother and more stable flow than always snapping to steepest descent
- connect this to the `erosion` function in `terrain_erosion.wgsl`

## Sediment capacity, erosion, and deposition

- define the intuition behind carrying capacity
- explain the capacity formula used in the shader in terms of

1. slope
2. speed
3. water
4. droplet size

- explain the three major branches in the implementation

1. if the droplet is near death, dump the remaining sediment
2. if the droplet is over capacity, deposit the excess
3. if the droplet is under capacity, erode more terrain

- discuss why erosion is clamped by the available local height difference
- explain why deposition uses a smooth radial distribution rather than a single cell write

## Why deposition is spread with a normal curve

- explain the visual problem with point deposition
- introduce the idea of using a radial falloff to distribute sediment
- discuss the normal distribution used in the shader
- explain why broad soft deposits create more natural landforms
- mention that this also helps avoid spiky artifacts

## The heightmap representation in the city builder

- explain how the terrain is stored as a flat `Vec<f32>`
- describe the world dimensions and how the 2d coordinates are mapped into a 1d array
- mention the normalized height range and later scaling by `WORLD_HEIGHT_SCALE`
- connect this representation to `src/world_gen/heightmap.rs`
- explain why a dense height buffer is a natural fit for both erosion and mesh generation

## The full terrain pipeline in the project

- explain the order of operations in the city builder

1. generate a base heightmap from procedural noise on the CPU
2. trigger the erosion stage once height generation is complete
3. upload the heightmap and droplets to GPU buffers
4. run multiple erosion batches
5. apply a blur pass
6. copy the final heightmap back into the world resource
7. rebuild the terrain mesh from the eroded heightmap

- connect this flow to `src/world_gen.rs`
- explain that erosion is not a standalone demo but part of the actual world generation process

## Why this is a good fit for the GPU

- motivate the GPU from a workload perspective instead of just saying it is faster
- explain that each droplet follows the same logic over different data
- discuss data parallelism and SIMD style execution
- explain that thousands of droplets can be processed simultaneously
- mention that the heightmap is shared state, which makes the problem interesting
- discuss why erosion is computationally expensive enough that CPU only execution becomes unattractive at large droplet counts

## A short theory section on the GPU

- explain the difference between the CPU and GPU execution models
- define a compute shader as a general purpose program run on the GPU
- explain invocations, workgroups, and dispatch size
- show how the total number of erosion jobs is `workgroup_size * dispatch_size`
- discuss why GPUs are good at throughput rather than low latency
- explain why branchy code is still acceptable here even if it is not perfectly ideal

## A short theory section on WebGPU and WGSL

- explain what WebGPU is at a high level
- describe WGSL as the shader language used for compute
- explain bind groups and storage buffers in practical terms
- introduce the host shader split

1. rust allocates and fills buffers
2. webgpu dispatches the shader
3. wgsl reads and mutates the buffers
4. rust reads the result back when needed

- mention why this API style is more explicit than older graphics APIs
- connect this section to the buffers and dispatch setup in `src/world_gen/erosion.rs`

## Synchronizing constants between Rust and WGSL

- explain the practical issue of keeping CPU side and shader side constants in sync
- show how the project generates `constants.wgsl` from Rust at startup
- discuss why this is useful for

1. image dimensions
2. workgroup sizes
3. erosion step count
4. blur configuration

- connect this to `src/shader_preprocessing.rs` and `assets/shaders/constants.wgsl`

## The Rust side compute worker

- explain the `ErosionComputeWorker` and what resources it owns
- describe the two main buffers

1. the droplet array
2. the heightmap results buffer

- explain the difference between staging and storage in the code
- discuss the one shot worker design and why the erosion is run in chunks over multiple frames
- mention deterministic seeding of droplets from the world seed

## Walking through the erosion shader

- break the shader discussion into small subsections

### Buffer layout and entry point

- explain the droplet storage array and the results height buffer
- explain why one invocation corresponds to one droplet

### Direction update

- show how the neighborhood loop accumulates a direction vector
- explain normalization and why the next position is rounded back onto the grid

### Water and speed update

- explain evaporation and the speed update from height difference
- discuss the intuition that steeper downhill paths accelerate the droplet

### Carry capacity and branching

- explain the carry capacity calculation carefully
- walk through the erosion and deposition branches

### Radius based deposit and erosion

- explain that `erode` is implemented as negative deposition
- discuss why this keeps the logic compact

## Parallelism and shared memory hazards

- explain that all droplets mutate the same heightmap buffer
- be honest that this means multiple invocations can target the same cell
- discuss the consequences

1. weak ordering
2. possible write conflicts
3. approximate rather than strictly deterministic evolution

- explain why this is still acceptable for terrain synthesis
- mention that visual plausibility matters more than exact reproducibility here
- discuss possible future improvements such as atomic accumulation or multi pass reduction schemes

## Why a blur pass comes after erosion

- explain that raw droplet erosion can leave sharp local artifacts on a discrete grid
- introduce the blur pass as a cheap smoothing step over the final heightmap
- discuss why this is not meant to be physically accurate water transport, but a practical cleanup pass
- connect this to `src/utils/blur.rs` and `assets/shaders/blur.wgsl`
- explain the visual tradeoff between smoothing artifacts and washing out fine details

## Turning the eroded heightmap into visible terrain

- explain how the final heightmap is fed into the mesh generation stage
- connect this to `src/world_gen/mesh_gen.rs`
- discuss how erosion affects

1. silhouette
2. slopes
3. valleys
4. normals
5. biome and texture placement indirectly through elevation changes

- emphasize that this is where the compute work becomes visible to the player

## Parameter tuning and artistic control

- discuss the parameters that shape the final terrain most strongly

1. droplet count
2. max erosion steps
3. droplet radius range
4. gravity
5. inertia
6. evaporation speed
7. erosion speed
8. deposition speed

- explain how changing each parameter affects the output terrain
- mention that there is always a balance between realism, stability, and preserving the original terrain shapes

## Limitations of the current implementation

- mention that the simulation is local and droplet based, not a full hydrology model
- note the lack of explicit river accumulation fields
- discuss grid discretization artifacts
- mention the approximate nature of concurrent writes
- discuss why the blur pass is both helpful and somewhat brute force
- mention that these limitations are often acceptable in a game terrain pipeline

## Future directions

- list some possible improvements to explore later

1. better gradient sampling instead of rounded movement
2. explicit flow map generation
3. separate accumulation and application passes
4. more stable parallel writes
5. multiple erosion material layers
6. coupling erosion to biome generation or road placement

- mention that the current system is already a strong practical middle ground between realism and performance

## Conclusion

- restate the main story of the article
- raw noise gives the terrain an initial shape
- erosion gives that terrain history
- the GPU makes it practical to run enough droplets to get convincing results
- webgpu and compute shaders make the implementation explicit but manageable
- close by pointing out that the final rendered terrain is really the result of both simulation and rendering working together

## Appendix ideas

- side by side screenshots of terrain before and after erosion
- a diagram of one droplet update step
- a short table of parameters and their visual effects
- selected code excerpts from the Rust worker and WGSL shader
- benchmarking notes on how long the erosion pass takes for different droplet counts
