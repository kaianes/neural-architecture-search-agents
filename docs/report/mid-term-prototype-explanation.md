#  Mid-term Prototype Explanation 
#### (Autonomous NAS, V0.1)

<p align="center">
  <img src="../figures/ANAS-logo.png" width="40%">
</p>


In this stage of the project, the project has a foundational system for an **autonomous Neural Architecture Search (NAS)** framework using Python.

The goal is to reduce human intervention in DNN design by enabling an AI-driven process that **automatically designs, trains, and evaluates deep neural network architectures**.

Additionallly, the project aims to minimize the computational cost of model search, aligning with **Green AI** principles that focus on sustainability, energy-efficient and eco-friendly AI research.

## What the prototype does
- AI-agent-based Neural Architecture Search (NAS) pipeline focused on accuracy, efficiency, and sustainability.
- Uses a small CNN baseline and Optuna to explore hyperparameters (channels, kernel size, dropout, learning rate).
- Tracks optional efficiency metrics (params, FLOPs) and can log carbon metrics *if CodeCarbon is installed.*
- Produces a summarized report of the search, with best trial and leaderboard.

## Agent pipeline (how it runs)
1) **CoordinatorAgent** orchestrates the run.  
2) **SearchAgent (Optuna)** samples hyperparameters and trains/evaluates SimpleCNN for a few epochs.  
3) **EvaluationAgent** summarizes results (best value/params, top trials, attrs).  
4) Summary is logged; a JSON summary is saved to `experiments/results/optuna_summary_<timestamp>.json`.

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
* A structured directory ready for multi-agent expansion (Coordinator, Search and Evaluation Agents).

> In essence, the system currently functions as an **automated experimental scientist**: it generates, tests, and evaluates neural architectures, learning which designs are most efficient in terms of accuracy, computation, and environmental impact.

##  What’s Happening During Execution

When you run the command:

```bash
python src/main.py --config experiments/configs/baseline.yaml
```

the following process unfolds:

### 1. **Configuration Loading**

The system reads the YAML configuration file (`baseline.yaml`) to get all experiment parameters—dataset type, training setup, search strategy, and file paths.

Optuna runs `n_trials` (from config), trains quick epochs, and evaluates accuracy.

### 2. **Environment Initialization**

The script sets a random seed for reproducibility and selects the best available device (CPU or GPU). All required directories for results and artifacts are created automatically.

### 3. **Neural Architecture Search (NAS) Begins**

Datasets (MNIST/Fashion-MNIST/CIFAR-10) auto-download to data/ on first run.

An AI agent (in this baseline, using **Optuna**) starts trying out different neural network configurations. For each trial:

* A **SimpleCNN** model is created with random hyperparameters (e.g., kernel size, channels, dropout rate).
* The model is trained for a few epochs on the chosen dataset (e.g., MNIST).
* Validation accuracy is measured and reported back to the NAS agent.

### 4. **Optimization and Selection**

Optuna evaluates all trials, tracking accuracy and computational metrics (like parameters and FLOPs). Once the trials finish, it identifies the **best model configuration**.

### 5. **Logging and Reporting**

Results—including accuracy, parameters, FLOPs, and energy consumption (if available)—are logged for analysis. The system outputs the best configuration and saves it for later use or further optimization saved to `experiments/results/optuna_summary_<timestamp>.json`

### Meaning of the Tools Used

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


## Files to point out (for the demo)
1. `src/main.py`: entrypoint; wires coordinator → search → evaluation.
2. `src/agents/`: coordinator_agent.py, search_agent.py, evaluation_agent.py.
3. `src/nas/optuna_search.py`: Optuna objective, trial loop, and result persistence.
4. `src/models/simple_cnn.py`: baseline CNN being searched.
5. `src/datasets/loader.py`: TorchVision loaders + normalization for MNIST/Fashion-MNIST/CIFAR-10.
6. `experiments/configs/baseline.yaml`: knobs for dataset, epochs, trials, paths.
7. `experiments/results/`: where the JSON summary is saved.

##  Why It Matters

* **Automation:** Reduces human workload in DNN design.
* **Efficiency:** Finds high-performing architectures with fewer experiments.
* **Sustainability:** Tracks and minimizes computational waste.
* **Scalability:** Framework can be extended to larger datasets or more complex NAS strategies.


| Aspect                      | Description                                                                                         |
| --------------------------- | --------------------------------------------------------------------------------------------------- |
| **Purpose**                 | Automatically discover new deep neural network (DNN) architectures efficiently and sustainably.     |
| **Input**                   | Image datasets (MNIST, Fashion-MNIST, CIFAR-10) and search parameters.                              |
| **Output**                  | Optimized neural network architecture with high accuracy and low computational cost.                |
| **Type of Intelligence**    | Multi-agent system with autonomous learning and coordinated decision-making.                        |
| **Learning Process**        | Experiment-driven: the system tests, evaluates, and improves based on obtained results.             |
| **Focus**                   | Computational efficiency, sustainability (low CO₂e), and automation of neural design.               |
| **Involved Agents**         | Coordinator Agent, Search Agent, Evaluation Agent.        |
| **Decision-Making Process** | Each agent has a defined role and shares information to collaboratively enhance system performance. |
| **Expected Outcome**        | An autonomous system capable of designing efficient and environmentally responsible AI models.      |

## Next Steps
* Expand search strategies to include Reinforcement Learning and Evolutionary Algorithms.
* Integrate more complex datasets (e.g., CIFAR-100, ImageNet).
* Enhance the agent communication protocol for better coordination.
* Implement a more sophisticated evaluation agent with deeper analysis capabilities.
* Cloud implementation.
* User interface for easier experiment configuration and monitoring.