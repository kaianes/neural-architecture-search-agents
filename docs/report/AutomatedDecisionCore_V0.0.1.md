# Automated decision core - V0.0.1

## Summary


In this stage of the project, we developed the **foundational system** for an **autonomous Neural Architecture Search (NAS)** framework using Python and PyTorch.
The goal is to enable an **AI-driven process that automatically designs, trains, and evaluates deep neural network architectures** with a focus on **efficiency and sustainability**.

The current implementation uses:

* **PyTorch** for building and training deep learning models (e.g., SimpleCNN).
* **Optuna** as the **search engine (Search Agent)** that automatically explores and optimizes model hyperparameters.
* **CodeCarbon** to track **energy consumption and CO₂ emissions**, ensuring compliance with **Green AI principles**.
* **TorchVision** for standard image datasets (MNIST, Fashion-MNIST, CIFAR-10).
* **Rich** and **YAML** for configuration, logging, and experiment organization.

The codebase is modular and already includes:

* A **search pipeline** (`optuna_search.py`) that runs multiple trials, evaluates models, and logs results.
* **Utility modules** for metrics, logging, environment setup, and dataset loading.
* **Carbon tracking integration** for sustainability assessment.
* A structured directory ready for multi-agent expansion (Coordinator, Evaluation, and Sustainability Agents).

In essence, the system currently functions as an **automated experimental scientist** — it generates, tests, and evaluates neural architectures, learning which designs are most efficient in terms of accuracy, computation, and environmental impact.

## Objective

The purpose of this project, **Efficient Deep Neural Network Architecture Search Using AI Agents**, is to create an intelligent system that can automatically design and optimize deep neural networks (DNNs). Instead of manually deciding the number of layers, neurons, or hyperparameters, AI agents will handle these tasks autonomously, improving both **efficiency** and **sustainability**.

The project aims to reduce human intervention in DNN design and minimize the computational cost of model search, aligning with **Green AI** principles that focus on energy-efficient and eco-friendly AI research.

---

##  What’s Happening During Execution

When you run the command:

```bash
python src/main.py --config experiments/configs/baseline.yaml
```

the following process unfolds:

### 1. **Configuration Loading**

The system reads the YAML configuration file (`baseline.yaml`) to get all experiment parameters—dataset type, training setup, search strategy, and file paths.

### 2. **Environment Initialization**

The script sets a random seed for reproducibility and selects the best available device (CPU or GPU). All required directories for results and artifacts are created automatically.

### 3. **Neural Architecture Search (NAS) Begins**

An AI agent (in this baseline, using **Optuna**) starts trying out different neural network configurations. For each trial:

* A **SimpleCNN** model is created with random hyperparameters (e.g., kernel size, channels, dropout rate).
* The model is trained for a few epochs on the chosen dataset (e.g., MNIST).
* Validation accuracy is measured and reported back to the NAS agent.

### 4. **Optimization and Selection**

Optuna evaluates all trials, tracking accuracy and computational metrics (like parameters and FLOPs). Once the trials finish, it identifies the **best model configuration**.

### 5. **Logging and Reporting**

Results—including accuracy, parameters, FLOPs, and energy consumption (if available)—are logged for analysis. The system outputs the best configuration and saves it for later use or further optimization.

---

## 🧠 Meaning of the Tools Used

| Tool / Library              | Purpose                     | Description                                                                                 |
| --------------------------- | --------------------------- | ------------------------------------------------------------------------------------------- |
| **PyTorch**                 | Deep Learning Framework     | Used to build, train, and evaluate neural network models (e.g., SimpleCNN).                 |
| **Optuna**                  | Hyperparameter Optimization | Conducts the automated search for optimal architectures by testing multiple configurations. |
| **TorchVision**             | Dataset & Model Utilities   | Provides built-in datasets (MNIST, CIFAR) and preprocessing tools.                          |
| **THOP**                    | FLOPs Calculator            | Estimates the computational complexity of a model by counting floating-point operations.    |
| **CodeCarbon** *(optional)* | Carbon Tracking             | Measures energy use and CO₂ emissions during training to promote sustainability.            |
| **Rich**                    | Console Logging             | Enhances terminal outputs with color and formatting for better readability.                 |
| **YAML**                    | Configuration Language      | Stores experiment parameters and paths in a readable format.                                |
| **TQDM**                    | Progress Bars               | Displays real-time training progress during each NAS trial.                                 |

---

## Simplified Flow Summary

```
Configuration → Dataset Loading → Agent (Optuna) → Model Trials → Evaluation → Best Architecture → Logging
```

1. **Load config:** reads YAML settings.
2. **Prepare dataset:** downloads and loads MNIST/CIFAR.
3. **Run trials:** Optuna explores different CNN designs.
4. **Evaluate:** measures accuracy and FLOPs.
5. **Select:** saves the best-performing design.

---

## 🌿 Why It Matters

* **Automation:** Reduces human workload in DNN design.
* **Efficiency:** Finds high-performing architectures with fewer experiments.
* **Sustainability:** Tracks and minimizes computational waste.
* **Scalability:** Framework can be extended to larger datasets or more complex NAS strategies.

---


| Aspect                      | Description                                                                                         |
| --------------------------- | --------------------------------------------------------------------------------------------------- |
| **Purpose**                 | Automatically discover new deep neural network (DNN) architectures efficiently and sustainably.     |
| **Input**                   | Image datasets (MNIST, Fashion-MNIST, CIFAR-10) and search parameters.                              |
| **Output**                  | Optimized neural network architecture with high accuracy and low computational cost.                |
| **Type of Intelligence**    | Multi-agent system with autonomous learning and coordinated decision-making.                        |
| **Learning Process**        | Experiment-driven: the system tests, evaluates, and improves based on obtained results.             |
| **Focus**                   | Computational efficiency, sustainability (low CO₂e), and automation of neural design.               |
| **Involved Agents**         | Search Agent, Evaluation Agent, Optimization Agent, Coordinator Agent, Sustainability Agent.        |
| **Decision-Making Process** | Each agent has a defined role and shares information to collaboratively enhance system performance. |
| **Expected Outcome**        | An autonomous system capable of designing efficient and environmentally responsible AI models.      |
