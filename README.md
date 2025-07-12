# 🌀 Diffusion-Based Particle Configuration Generator

A physics-informed deep generative model for synthesizing particle configurations in accelerator physics using **denoising diffusion probabilistic models (DDPMs)**. This framework generates physically plausible particle arrangements by modeling collective electromagnetic effects, especially Coulomb interactions, conditioned on beam energy.

---

## 📌 Key Features

- 🧠 **Conditional Diffusion Model** for generating 2D particle coordinates
- ⚡ **Coulomb Force Simulation** captures repulsion & clustering effects
- 🔬 **Beam Energy Conditioning** for realistic energy-dependent distributions
- 🧪 Physics-informed force fields during data generation
- 🔁 Full diffusion loop: training, noise injection, reverse sampling

---

## 📐 Technical Highlights

### 🧮 Physics Integration
- **Coulomb Force Law** implemented in O(N²) for all particle interactions
- Charge regularization avoids singularities at close ranges
- Beam-like Gaussian distributions simulated in initial state

### 🧊 Diffusion Model
- Linear beta noise schedule (0.0001 → 0.02)
- Noise prediction via MSE loss
- 3-layer fully-connected network with ReLU activations
- Conditioning via timestep + beam energy

### 🛠 Code Structure

- `ParticleDataGenerator`: Creates synthetic datasets with collective effects
- `SimpleDiffusionModel`: Learns to predict denoising vectors
- `DiffusionTrainer`: Trains the model, samples new particle configurations

---

## 🧪 Sample Workflow

#### python
### Generate particle config with collective effects
positions, charges, energy = generator.generate_configuration(beam_energy=1.2, charge_ratio=0.5)

### Train model
loss = trainer.train_step(positions, conditions, optimizer)

### Sample configurations
samples = trainer.sample(n_samples=64, condition=torch.tensor([1.2]), shape=(n_particles, 2))

## 📊 Input/Output Shapes

### 🧠 Training Phase
#### python
#### Input tensors
- positions.shape      (batch_size, n_particles, 2)
- conditions.shape     (batch_size, 1)
- timesteps.shape      (batch_size, 1)

### Output tensor
- predicted_noise.shape   (batch_size, n_particles, 2)

### References 
[1] Ho et al., “Denoising Diffusion Probabilistic Models,” 2020.  
    📄 https://arxiv.org/abs/2006.11239

[2] Dhariwal & Nichol, “Diffusion Models Beat GANs on Image Synthesis,” 2021.  
    📄 https://arxiv.org/abs/2105.05233

[3] Raissi et al., “Physics-informed neural networks,” 2019.  
    📄 https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125

[4] Wiedemann, “Particle Accelerator Physics,” Springer, 2007.

[5] Chao & Tigner, “Handbook of Accelerator Physics and Engineering,” 2013.

[6] Qiang et al., “An object-oriented parallel particle-in-cell code for beam dynamics simulation,” 2006.

[7] Qi et al., “PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation,” 2017.  
    📄 https://arxiv.org/abs/1612.00593

[8] Battaglia et al., “Relational inductive biases, deep learning, and graph networks,” 2018.  
    📄 https://arxiv.org/abs/1806.01261



  ## 🪪 License
- This project is licensed under the [MIT License](LICENSE).
  
