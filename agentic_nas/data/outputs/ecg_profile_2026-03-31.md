{
  "profile_summary": {
    "data_modality": "1D Time-Series (ECG)",
    "likely_task_type": "Binary Classification",
    "key_constraints": [
      "Small sample size (2,660) relative to feature count (2,500)",
      "High risk of overfitting due to 1:1 sample-to-feature ratio",
      "Input requires reshaping from 1D vector to 2D tensor (length, channels)"
    ]
  },
  "selected_model_families": [
    {
      "family": "cnn1d",
      "rationale": "Well-established for physiological signals (ECG); spatial filters effectively capture local morphological patterns in waveforms while sharing weights to mitigate overfitting.",
      "priority": "high"
    },
    {
      "family": "mlp",
      "rationale": "Provides a robust baseline for low-sample tabular-style data; lower complexity than temporal models like RNNs which may struggle with 2,500 time steps.",
      "priority": "medium"
    }
  ],
  "search_space": {
    "cnn1d": {
      "hyperparameters": {
        "num_blocks": {
          "type": "int",
          "min": 1,
          "max": 3
        },
        "filters": {
          "type": "categorical",
          "values": [16, 32, 64]
        },
        "kernel_size": {
          "type": "categorical",
          "values": [3, 7, 11]
        },
        "pooling_type": {
          "type": "categorical",
          "values": ["max", "avg"]
        },
        "dropout_rate": {
          "type": "float",
          "min": 0.2,
          "max": 0.5,
          "step": 0.1
        },
        "learning_rate": {
          "type": "float",
          "min": 0.0001,
          "max": 0.01,
          "log": true
        }
      }
    },
    "mlp": {
      "hyperparameters": {
        "num_layers": {
          "type": "int",
          "min": 1,
          "max": 2
        },
        "hidden_units": {
          "type": "categorical",
          "values": [128, 256, 512]
        },
        "dropout_rate": {
          "type": "float",
          "min": 0.2,
          "max": 0.5,
          "step": 0.1
        },
        "activation": {
          "type": "categorical",
          "values": ["relu", "selu"]
        }
      }
    }
  },
  "uncertainty_notes": [
    "Sampling frequency is unknown, making it difficult to optimize kernel sizes for specific physiological events (e.g., QRS complex).",
    "Absence of patient IDs makes it unclear if samples are i.i.d. or require GroupKFold validation.",
    "Lack of signal units (raw vs normalized) makes weight initialization sensitivity uncertain."
  ],
  "handoff_notes": [
    "Reshape input from (2500,) to (2500, 1) before passing to cnn1d.",
    "Mandatory removal of 56 duplicate rows prior to training.",
    "Use Stratified 5-Fold Cross-Validation due to the limited sample size (N=2,660).",
    "Strong L2 regularization or weight decay is recommended given the feature-to-sample ratio."
  ]
}